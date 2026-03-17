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
from ..database.models import Speech, Term, ModelVersion, TrumpEvent
from .markov_trainer import MarkovChainTrainer
from .colab_integration import ColabPredictor
from ..config import config

MIN_NEW_SPEECHES_FOR_RETRAIN = 5


class LocalPipeline:
    """Orchestrates local training and prediction pipeline."""

    def __init__(self):
        self.trainer = MarkovChainTrainer(order=config.markov_order)
        self.colab_predictor = ColabPredictor()
        self.fine_tuner = None  # lazy-loaded
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

    def _get_fine_tuner(self):
        """Lazy-load GPT2FineTuner only when needed."""
        if self.fine_tuner is None:
            try:
                from .fine_tuner import GPT2FineTuner
                self.fine_tuner = GPT2FineTuner()
            except ImportError:
                logger.debug("Fine-tuner dependencies not installed")
                return None
        return self.fine_tuner

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

        # Include fine-tune status when available
        ft = self._get_fine_tuner()
        if ft:
            ft_status = ft.get_status()
            if ft_status['state'] != 'idle':
                status['fine_tune_status'] = ft_status

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

            # Phase 3: Run Monte Carlo (2C: per-event scenario weighting)
            scenario_weights = self._get_event_scenario_weights()
            self._status.update(stage='Running Monte Carlo', progress=0.3)
            weight_info = f' (scenario weights: {scenario_weights})' if scenario_weights else ''
            self._log_event(f'Phase 3: Running {config.monte_carlo_simulations} Monte Carlo simulations...{weight_info}')
            predictions_data = self.trainer.run_monte_carlo(
                terms=term_list,
                num_simulations=config.monte_carlo_simulations,
                scenario_weights=scenario_weights,
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

            # Update status — Markov pipeline complete
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

            # Phase 6-8: GPT-2 fine-tuning (non-blocking — runs in background)
            if config.fine_tune_enabled:
                self._log_event('Phase 6: Starting GPT-2 fine-tuning in background...')
                import threading
                ft_thread = threading.Thread(
                    target=self._run_fine_tune_phases,
                    args=(term_list,),
                    daemon=True,
                )
                ft_thread.start()

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

    def _get_event_scenario_weights(self) -> Optional[dict]:
        """2C: Compute scenario weights based on the next known Trump event.

        If the next event is a rally, heavily weight rally simulations.
        Falls back to None (use default weights) if no upcoming event.
        """
        EVENT_TYPE_MAP = {
            'rally': {'rally': 0.85, 'press_conference': 0.05, 'chopper_talk': 0.03, 'fox_interview': 0.05, 'social_media': 0.02},
            'press_conference': {'rally': 0.05, 'press_conference': 0.80, 'chopper_talk': 0.05, 'fox_interview': 0.05, 'social_media': 0.05},
            'interview': {'rally': 0.05, 'press_conference': 0.05, 'chopper_talk': 0.05, 'fox_interview': 0.80, 'social_media': 0.05},
            'fox_interview': {'rally': 0.05, 'press_conference': 0.05, 'chopper_talk': 0.05, 'fox_interview': 0.80, 'social_media': 0.05},
            'state_dinner': {'rally': 0.10, 'press_conference': 0.60, 'chopper_talk': 0.10, 'fox_interview': 0.10, 'social_media': 0.10},
            'signing_ceremony': {'rally': 0.10, 'press_conference': 0.60, 'chopper_talk': 0.10, 'fox_interview': 0.10, 'social_media': 0.10},
        }

        try:
            with get_session() as session:
                next_event = session.query(TrumpEvent).filter(
                    TrumpEvent.start_time >= datetime.utcnow(),
                    TrumpEvent.is_confirmed == True,
                ).order_by(TrumpEvent.start_time).first()

                if next_event and next_event.event_type:
                    event_type = next_event.event_type.lower().strip()
                    weights = EVENT_TYPE_MAP.get(event_type)
                    if weights:
                        logger.info(f"2C: Next event is '{event_type}' — adjusting scenario weights")
                        return weights
        except Exception as e:
            logger.debug(f"Could not get event scenario weights: {e}")

        return None

    def _run_fine_tune_phases(self, term_list: list[str]):
        """Phase 6-8: GPT-2 fine-tuning + Monte Carlo (runs in background thread)."""
        ft = self._get_fine_tuner()
        if not ft:
            self._log_event('Phase 6: Skipped — fine-tuner dependencies not installed')
            return

        try:
            # Phase 6: Fine-tune GPT-2
            self._log_event('Phase 6: Fine-tuning GPT-2 with LoRA (this will take hours)...')
            ft_result = ft.train()
            if not ft_result or ft_result.get('status') == 'error':
                self._log_event(f'Phase 6: Fine-tuning failed: {ft_result}')
                return
            self._log_event(f'Phase 6: Fine-tuning complete — {ft_result.get("total_steps", 0)} steps, loss={ft_result.get("final_loss", "?")}')

            # Phase 7: GPT-2 Monte Carlo
            if term_list:
                self._log_event(f'Phase 7: Running GPT-2 Monte Carlo ({config.fine_tune_mc_sims} sims)...')
                gpt2_predictions = ft.run_monte_carlo(term_list)
                if gpt2_predictions and gpt2_predictions.get('term_predictions'):
                    self._log_event(f'Phase 7: GPT-2 Monte Carlo complete — {len(gpt2_predictions["term_predictions"])} predictions')

                    # Phase 8: Blend with Markov predictions
                    self._blend_predictions(gpt2_predictions)
                    self._log_event('Phase 8: Blended GPT-2 + Markov predictions')
                else:
                    self._log_event('Phase 7: GPT-2 Monte Carlo produced no predictions')

        except Exception as e:
            self._log_event(f'Fine-tune pipeline error: {e}')
            logger.error(f"Fine-tune phases failed: {e}")

    def _blend_predictions(self, gpt2_predictions: dict):
        """Blend GPT-2 Monte Carlo predictions with existing Markov predictions.

        Saves a blended predictions file with weighted average (60% Markov, 40% GPT-2).
        """
        import json

        pred_path = os.path.join('data', 'predictions', 'predictions_latest.json')
        if not os.path.exists(pred_path):
            return

        with open(pred_path) as f:
            markov_data = json.load(f)

        markov_preds = {p['term']: p for p in markov_data.get('term_predictions', [])}
        gpt2_preds = {p['term']: p for p in gpt2_predictions.get('term_predictions', [])}

        blended = []
        for term, mp in markov_preds.items():
            gp = gpt2_preds.get(term)
            if gp:
                # Weighted blend: 60% Markov (more sims, proven), 40% GPT-2
                blended_prob = 0.6 * mp['probability'] + 0.4 * gp['probability']
                entry = mp.copy()
                entry['probability'] = round(blended_prob, 4)
                entry['model_name'] = 'blended_markov_gpt2'
                entry['markov_probability'] = mp['probability']
                entry['gpt2_probability'] = gp['probability']
            else:
                entry = mp.copy()
            blended.append(entry)

        markov_data['term_predictions'] = blended
        markov_data['blended'] = True
        markov_data['blend_weights'] = {'markov': 0.6, 'gpt2': 0.4}

        with open(pred_path, 'w') as f:
            json.dump(markov_data, f, indent=2)

        # Re-import to DB
        self.colab_predictor.save_to_database()

    def run_fine_tune_only(self):
        """Manually trigger fine-tuning without running the Markov pipeline."""
        ft = self._get_fine_tuner()
        if not ft:
            return {'status': 'error', 'error': 'Fine-tuner dependencies not installed'}

        self._log_event('Manual fine-tune triggered (skipping Markov phases)')
        import threading
        thread = threading.Thread(target=ft.train, daemon=True)
        thread.start()
        return {'status': 'started'}

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
