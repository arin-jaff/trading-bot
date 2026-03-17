"""Local training pipeline for Raspberry Pi.

Replaces ColabPipeline with a fully local flow:
scrape -> train Markov chain -> Monte Carlo simulate -> predict -> save to DB.

No Google Drive, no Colab, no GPU required.
"""

import threading
import time
from datetime import datetime
from typing import Optional
from loguru import logger

from ..database.db import get_session
from ..database.models import Speech, Term, ModelVersion
from .markov_trainer import MarkovChainTrainer
from .colab_integration import ColabPredictor
from ..config import config

MIN_NEW_SPEECHES_FOR_RETRAIN = 5


class LocalPipeline:
    """Orchestrates local training and prediction pipeline."""

    def __init__(self):
        self.trainer = MarkovChainTrainer(order=config.markov_order)
        self.colab_predictor = ColabPredictor()
        self._lock = threading.Lock()
        self._log = []  # list of {timestamp, message} dicts
        self._max_log = 200
        self._status = {
            'state': 'idle',  # idle, running, complete, error
            'stage': '',
            'progress': 0.0,
            'started_at': None,
            'completed_at': None,
            'last_training_speech_count': self._get_last_training_count(),
            'current_version': self._get_current_version(),
            'error': None,
        }

    def get_status(self) -> dict:
        """Get pipeline status including trainer progress."""
        status = self._status.copy()

        # Compute elapsed time
        if status.get('started_at'):
            try:
                started = datetime.fromisoformat(status['started_at'])
                elapsed = (datetime.utcnow() - started).total_seconds()
                status['elapsed_seconds'] = round(elapsed, 1)
            except Exception:
                status['elapsed_seconds'] = None
        else:
            status['elapsed_seconds'] = None

        # Merge in trainer-level progress when running
        if status['state'] == 'running':
            trainer_status = self.trainer.get_status()
            status['trainer'] = trainer_status

            # If trainer is simulating, use its detailed progress
            if trainer_status['state'] == 'simulating':
                status['stage'] = (
                    f"Monte Carlo: {trainer_status['current_simulation']}"
                    f"/{trainer_status['total_simulations']}"
                )
                status['progress'] = 0.3 + 0.6 * trainer_status['progress']

            # Compute overall ETA from progress
            progress = status.get('progress', 0)
            elapsed = status.get('elapsed_seconds')
            if progress > 0.01 and elapsed:
                total_estimated = elapsed / progress
                status['eta_seconds'] = round(total_estimated - elapsed, 1)
            elif trainer_status.get('eta_seconds'):
                status['eta_seconds'] = trainer_status['eta_seconds']
            else:
                status['eta_seconds'] = None
        else:
            status['eta_seconds'] = None

        return status

    def _log_event(self, message: str):
        """Add a timestamped entry to the pipeline log."""
        self._log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'message': message,
        })
        if len(self._log) > self._max_log:
            self._log = self._log[-self._max_log:]

    def get_log(self, limit: int = 50) -> list[dict]:
        """Get recent pipeline log entries."""
        return list(reversed(self._log[-limit:]))

    def should_retrain(self) -> bool:
        """Check if enough new data has accumulated to justify retraining."""
        with get_session() as session:
            current_count = session.query(Speech).filter(
                Speech.transcript.isnot(None),
                Speech.is_processed == True,
                Speech.word_count >= 100,
            ).count()

        last_count = self._status.get('last_training_speech_count', 0)
        new_speeches = current_count - last_count

        if new_speeches >= MIN_NEW_SPEECHES_FOR_RETRAIN:
            logger.info(f"{new_speeches} new speeches since last training — retraining")
            return True

        # Also retrain if no model exists
        if not self._get_current_version():
            logger.info("No trained model found — training from scratch")
            return True

        return False

    def run_full_pipeline(self) -> dict:
        """Run the complete local pipeline: train -> simulate -> predict -> save."""
        if not self._lock.acquire(blocking=False):
            return {'status': 'already_running'}

        try:
            self._status.update(
                state='running',
                stage='Checking data',
                progress=0.0,
                started_at=datetime.utcnow().isoformat(),
                error=None,
            )
            self._log_event('Pipeline started')

            # Check if retraining is needed
            if not self.should_retrain():
                self._status.update(state='idle', stage='No retraining needed')
                self._log_event('Skipped — not enough new data')
                return {'status': 'skipped', 'reason': 'not enough new data'}

            # Phase 1: Train Markov chain
            self._status.update(stage='Training Markov chain', progress=0.1)
            self._log_event('Phase 1: Training Markov chain on speech corpus...')
            train_result = self.trainer.train()
            if not train_result:
                raise RuntimeError("Markov chain training failed")
            self._log_event(f'Markov chain trained: {train_result["corpus_size"]} speeches, {train_result["chain_states"]} states in {train_result["training_seconds"]:.1f}s')

            # Phase 2: Get terms to predict
            self._status.update(stage='Loading terms', progress=0.25)
            with get_session() as session:
                terms = session.query(Term).all()
                term_list = [t.normalized_term for t in terms]

            if not term_list:
                logger.warning("No tracked terms found")
                self._status.update(state='complete', stage='No terms to predict')
                self._log_event('No tracked terms found — nothing to predict')
                return {'status': 'complete', 'reason': 'no terms'}

            self._log_event(f'Phase 2: Loaded {len(term_list)} terms to predict')

            # Phase 3: Run Monte Carlo
            self._status.update(stage='Running Monte Carlo', progress=0.3)
            self._log_event(f'Phase 3: Running {config.monte_carlo_simulations} Monte Carlo simulations...')
            predictions_data = self.trainer.run_monte_carlo(
                terms=term_list,
                num_simulations=config.monte_carlo_simulations,
            )

            if not predictions_data or not predictions_data.get('term_predictions'):
                raise RuntimeError("Monte Carlo simulation produced no predictions")
            self._log_event(f'Monte Carlo complete: {len(predictions_data["term_predictions"])} term predictions generated')

            # Phase 4: Save predictions JSON
            self._status.update(stage='Saving predictions', progress=0.9)
            self.trainer.save_predictions(predictions_data)
            self._log_event('Phase 4: Predictions saved to data/predictions/')

            # Phase 5: Import into database
            self._status.update(stage='Importing to database', progress=0.95)
            self.colab_predictor.save_to_database()
            self._log_event('Phase 5: Predictions imported to database')

            # Update model version info in DB
            pred_count = len(predictions_data['term_predictions'])
            with get_session() as session:
                active_model = session.query(ModelVersion).filter_by(
                    is_active=True
                ).order_by(ModelVersion.created_at.desc()).first()
                if active_model:
                    active_model.simulation_count = config.monte_carlo_simulations
                    active_model.prediction_count = pred_count

            # Record speech count for should_retrain check
            with get_session() as session:
                current_count = session.query(Speech).filter(
                    Speech.transcript.isnot(None),
                    Speech.is_processed == True,
                    Speech.word_count >= 100,
                ).count()
            self._status['last_training_speech_count'] = current_count

            # Update status
            version = train_result['version']
            self._status.update(
                state='complete',
                stage=f'Pipeline complete — TrumpGPT v{version}',
                progress=1.0,
                completed_at=datetime.utcnow().isoformat(),
                current_version=version,
            )

            self._log_event(f'Pipeline complete — TrumpGPT v{version} ({pred_count} predictions from {train_result["corpus_size"]} speeches)')

            # Email notification
            self._notify_completion(version, train_result, pred_count)

            result = {
                'status': 'complete',
                'version': version,
                'corpus_size': train_result['corpus_size'],
                'predictions': pred_count,
                'training_seconds': train_result['training_seconds'],
            }
            logger.info(f"Local pipeline complete: {result}")
            return result

        except Exception as e:
            self._status.update(state='error', error=str(e))
            self._log_event(f'ERROR: {e}')
            logger.error(f"Local pipeline failed: {e}")
            self._notify_failure(str(e))
            return {'status': 'error', 'error': str(e)}

        finally:
            self._lock.release()

    def run_pipeline_async(self):
        """Run pipeline in a background thread."""
        thread = threading.Thread(target=self.run_full_pipeline, daemon=True)
        thread.start()
        return {'status': 'started'}

    def _get_last_training_count(self) -> int:
        """Get speech count from the last training run."""
        try:
            with get_session() as session:
                latest = session.query(ModelVersion).filter_by(
                    is_active=True
                ).order_by(ModelVersion.created_at.desc()).first()
                return latest.corpus_size if latest else 0
        except Exception:
            return 0

    def _get_current_version(self) -> Optional[str]:
        """Get the current active model version string."""
        try:
            with get_session() as session:
                latest = session.query(ModelVersion).filter_by(
                    is_active=True
                ).order_by(ModelVersion.created_at.desc()).first()
                return latest.version if latest else None
        except Exception:
            return None

    def _notify_completion(self, version: str, train_result: dict,
                           pred_count: int):
        """Send email notification on successful training."""
        try:
            from ..notifications.email_notifier import email_notifier
            if email_notifier.enabled:
                email_notifier.send_critical_alert(
                    f'TrumpGPT v{version} Trained',
                    f'New model trained on {train_result["corpus_size"]} speeches. '
                    f'{pred_count} term predictions generated.',
                    {
                        'Version': version,
                        'Corpus': f'{train_result["corpus_size"]} speeches',
                        'Training Time': f'{train_result["training_seconds"]:.1f}s',
                        'Predictions': pred_count,
                        'Chain States': train_result['chain_states'],
                    }
                )
        except Exception as e:
            logger.debug(f"Training notification email failed: {e}")

    def _notify_failure(self, error: str):
        """Send email notification on pipeline failure."""
        try:
            from ..notifications.email_notifier import email_notifier
            if email_notifier.enabled:
                email_notifier.send_critical_alert(
                    'Training Pipeline Failed',
                    f'The local training pipeline encountered an error: {error}',
                )
        except Exception as e:
            logger.debug(f"Failure notification email failed: {e}")
