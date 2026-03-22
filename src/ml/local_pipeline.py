"""Local training pipeline for Raspberry Pi.

Runs the Markov chain pipeline (Phases 0-5) every 6 hours.
Fine-tuning runs nightly on the Pi itself (Pythia-160M with LoRA,
~75 min on Pi 4). Fine-tuned predictions are automatically blended
with Markov predictions.
"""

import json
import os
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

# Path where Mac deposits Pythia predictions after fine-tuning
PYTHIA_PREDICTIONS_PATH = os.path.join('data', 'predictions', 'predictions_pythia.json')


class LocalPipeline:
    """Orchestrates the local Markov training pipeline.

    Phases 0-5 run every 6 hours (~35 seconds):
      0. Social media refresh (Twitter import + Truth Social scrape)
      1. Train Markov chain
      2. Load terms
      3. Monte Carlo simulation (2,000 sims)
      4. Save predictions
      5. Import to DB + blend with Pythia predictions if available
    """

    def __init__(self):
        self.trainer = MarkovChainTrainer(order=config.markov_order)
        self.colab_predictor = ColabPredictor()
        self._lock = threading.Lock()
        self._log = []
        self._max_log = 200
        self._status = {
            'state': 'idle',
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

        if status.get('started_at'):
            try:
                started = datetime.fromisoformat(status['started_at'])
                elapsed = (datetime.now() - started).total_seconds()
                status['elapsed_seconds'] = round(elapsed, 1)
            except Exception:
                status['elapsed_seconds'] = None
        else:
            status['elapsed_seconds'] = None

        if status['state'] == 'running':
            trainer_status = self.trainer.get_status()
            status['trainer'] = trainer_status

            if trainer_status['state'] == 'simulating':
                status['stage'] = (
                    f"Monte Carlo: {trainer_status['current_simulation']}"
                    f"/{trainer_status['total_simulations']}"
                )
                status['progress'] = 0.3 + 0.6 * trainer_status['progress']

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

        # Show whether Pythia predictions are available for blending
        status['pythia_predictions_available'] = os.path.exists(PYTHIA_PREDICTIONS_PATH)

        return status

    def _log_event(self, message: str):
        self._log.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
        })
        if len(self._log) > self._max_log:
            self._log = self._log[-self._max_log:]

    def get_log(self, limit: int = 50) -> list[dict]:
        return list(reversed(self._log[-limit:]))

    def should_retrain(self) -> bool:
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

        if not self._get_current_version():
            logger.info("No trained model found — training from scratch")
            return True

        # Force retrain if the Markov pickle is missing/corrupt
        if not self.trainer.chain:
            self.trainer._load_latest_model()
            if not self.trainer.chain:
                logger.warning("Markov model missing or corrupt — forcing retrain")
                return True

        return False

    def run_full_pipeline(self) -> dict:
        """Run Phases 0-5: scrape → train → simulate → predict → save."""
        if not self._lock.acquire(blocking=False):
            return {'status': 'already_running'}

        try:
            self._status.update(
                state='running', stage='Checking data', progress=0.0,
                started_at=datetime.now().isoformat(), error=None,
            )
            self._log_event('Pipeline started')

            # Phase 0: Social media refresh
            self._refresh_social_media()

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

            # Phase 2: Get terms
            self._status.update(stage='Loading terms', progress=0.25)
            with get_session() as session:
                terms = session.query(Term).all()
                term_list = [t.normalized_term for t in terms]

            if not term_list:
                self._status.update(state='complete', stage='No terms to predict')
                self._log_event('No tracked terms found — nothing to predict')
                return {'status': 'complete', 'reason': 'no terms'}

            self._log_event(f'Phase 2: Loaded {len(term_list)} terms to predict')

            # Phase 3: Monte Carlo
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

            # Phase 4: Save predictions
            self._status.update(stage='Saving predictions', progress=0.9)
            self.trainer.save_predictions(predictions_data)
            self._log_event('Phase 4: Predictions saved to data/predictions/')

            # Phase 5: Import to DB + blend with Pythia if available
            self._status.update(stage='Importing to database', progress=0.93)
            self.colab_predictor.save_to_database()
            self._log_event('Phase 5: Predictions imported to database')

            # Blend with Pythia predictions if they exist on disk
            blended = self._blend_pythia_if_available(predictions_data)
            if blended:
                self._log_event('Phase 5: Blended with Pythia predictions (60% Markov / 40% Pythia)')

            # Update model version
            pred_count = len(predictions_data['term_predictions'])
            with get_session() as session:
                active_model = session.query(ModelVersion).filter_by(
                    is_active=True
                ).order_by(ModelVersion.created_at.desc()).first()
                if active_model:
                    active_model.simulation_count = config.monte_carlo_simulations
                    active_model.prediction_count = pred_count

            # Record speech count
            with get_session() as session:
                current_count = session.query(Speech).filter(
                    Speech.transcript.isnot(None),
                    Speech.is_processed == True,
                    Speech.word_count >= 100,
                ).count()
            self._status['last_training_speech_count'] = current_count

            version = train_result['version']
            self._status.update(
                state='complete',
                stage=f'Pipeline complete — TrumpGPT v{version}',
                progress=1.0,
                completed_at=datetime.now().isoformat(),
                current_version=version,
            )
            self._log_event(f'Pipeline complete — TrumpGPT v{version} ({pred_count} predictions from {train_result["corpus_size"]} speeches)')
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
        thread = threading.Thread(target=self.run_full_pipeline, daemon=True)
        thread.start()
        return {'status': 'started'}

    # ── Fine-tuning on Pi ──

    def run_fine_tuning(self) -> Optional[dict]:
        """Run Pi-native fine-tuning (Pythia-160M with LoRA).

        Called nightly by the scheduler. Trains the model, runs Monte Carlo
        simulations, and saves predictions for blending with Markov output.
        Returns result dict or None on failure.
        """
        if not config.fine_tune_enabled:
            self._log_event('Fine-tuning disabled (FINE_TUNE_ENABLED=false)')
            return None

        try:
            from .fine_tuner import GPT2FineTuner
        except ImportError as e:
            self._log_event(f'Fine-tuning skipped: missing deps ({e})')
            logger.warning(f"Fine-tuning deps not available: {e}")
            return None

        self._log_event('Fine-tuning: starting Pi-native Pythia LoRA training...')
        self._status.update(stage='Fine-tuning Pythia (LoRA)', progress=0.0)

        try:
            fine_tuner = GPT2FineTuner()
            train_result = fine_tuner.train()

            if not train_result or train_result.get('status') in ('error', None):
                self._log_event(f'Fine-tuning failed: {train_result}')
                return None

            if train_result.get('status') == 'stopped':
                self._log_event('Fine-tuning stopped by user')
                return train_result

            self._log_event(
                f'Fine-tuning complete: v{train_result.get("version", "?")} '
                f'({train_result.get("total_steps", 0)} steps, '
                f'loss={train_result.get("final_loss", "?")}, '
                f'{train_result.get("training_seconds", 0)/60:.1f} min)'
            )

            # Run Monte Carlo with the fine-tuned model
            with get_session() as session:
                terms = session.query(Term).all()
                term_list = [t.normalized_term for t in terms]

            if term_list and fine_tuner.has_trained_model():
                self._log_event(f'Fine-tune MC: running {config.fine_tune_mc_sims} simulations...')
                mc_data = fine_tuner.run_monte_carlo(term_list)

                if mc_data and mc_data.get('term_predictions'):
                    # Save as predictions_pythia.json for blending
                    pred_path = PYTHIA_PREDICTIONS_PATH
                    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                    with open(pred_path, 'w') as f:
                        json.dump(mc_data, f, indent=2)
                    self._log_event(
                        f'Fine-tune MC: {len(mc_data["term_predictions"])} '
                        f'term predictions saved to {pred_path}'
                    )

            return train_result

        except Exception as e:
            self._log_event(f'Fine-tuning error: {e}')
            logger.error(f"Pi fine-tuning failed: {e}")
            return None

    # ── Pythia blending ──

    def _blend_pythia_if_available(self, markov_data: dict) -> bool:
        """Blend Pythia predictions from disk if predictions_pythia.json exists.

        This file is produced by the nightly Pi fine-tuning job or
        synced from Mac. Returns True if blending occurred.
        """
        if not os.path.exists(PYTHIA_PREDICTIONS_PATH):
            return False

        # Skip if predictions are older than 7 days
        max_age_days = 7
        file_age = time.time() - os.path.getmtime(PYTHIA_PREDICTIONS_PATH)
        if file_age > max_age_days * 86400:
            logger.warning(f"Pythia predictions are {file_age/86400:.0f} days old (>{max_age_days}d) — skipping blend")
            return False

        try:
            with open(PYTHIA_PREDICTIONS_PATH) as f:
                pythia_data = json.load(f)

            pythia_preds = {p['term']: p for p in pythia_data.get('term_predictions', [])}
            if not pythia_preds:
                return False

            markov_preds = {p['term']: p for p in markov_data.get('term_predictions', [])}

            blended = []
            for term, mp in markov_preds.items():
                pp = pythia_preds.get(term)
                if pp:
                    blended_prob = 0.6 * mp['probability'] + 0.4 * pp['probability']
                    entry = mp.copy()
                    entry['probability'] = round(blended_prob, 4)
                    entry['model_name'] = 'blended_markov_pythia'
                    entry['markov_probability'] = mp['probability']
                    entry['pythia_probability'] = pp['probability']
                else:
                    entry = mp.copy()
                blended.append(entry)

            markov_data['term_predictions'] = blended
            markov_data['blended'] = True
            markov_data['blend_weights'] = {'markov': 0.6, 'pythia': 0.4}

            # Re-save and re-import
            pred_path = os.path.join('data', 'predictions', 'predictions_latest.json')
            with open(pred_path, 'w') as f:
                json.dump(markov_data, f, indent=2)
            self.colab_predictor.save_to_database()
            return True

        except Exception as e:
            logger.warning(f"Pythia blend failed: {e}")
            return False

    # ── Social media ──

    def _refresh_social_media(self):
        """Phase 0: Import Twitter archive (one-time) + scrape Truth Social + Twitter/X."""
        try:
            from ..scraper.social_media_importer import SocialMediaImporter
            importer = SocialMediaImporter()

            init_result = importer.ensure_initial_import()
            if init_result.get('status') == 'imported':
                self._log_event(
                    f'Phase 0: Initial Twitter import — '
                    f'{init_result.get("imported", 0)} tweets, '
                    f'{init_result.get("daily_digests_created", 0)} daily digests'
                )

            new_posts = importer.scrape_latest_posts()
            if new_posts:
                self._log_event(f'Phase 0: Scraped {new_posts} new social media posts + rebuilt daily digests')

        except Exception as e:
            logger.warning(f"Social media refresh failed: {e}")
            self._log_event(f'Phase 0: Social media refresh failed (non-fatal): {e}')

    # ── Helpers ──

    def _get_last_training_count(self) -> int:
        try:
            with get_session() as session:
                latest = session.query(ModelVersion).filter_by(
                    is_active=True
                ).order_by(ModelVersion.created_at.desc()).first()
                return latest.corpus_size if latest else 0
        except Exception:
            return 0

    def _get_current_version(self) -> Optional[str]:
        try:
            with get_session() as session:
                latest = session.query(ModelVersion).filter_by(
                    is_active=True
                ).order_by(ModelVersion.created_at.desc()).first()
                return latest.version if latest else None
        except Exception:
            return None

    def _notify_completion(self, version, train_result, pred_count):
        """Log completion (email limited to daily digest only)."""
        logger.info(f"Pipeline complete: v{version}, {pred_count} predictions from {train_result['corpus_size']} speeches")

    def _get_event_scenario_weights(self) -> Optional[dict]:
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
                    TrumpEvent.start_time >= datetime.now(),
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

    def _notify_failure(self, error):
        """Log failure (email limited to daily digest only)."""
        logger.error(f"Pipeline failed: {error}")
