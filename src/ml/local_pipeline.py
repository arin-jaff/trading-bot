"""Local training pipeline for Raspberry Pi.

Unified 8-phase pipeline that runs automatically:
  Phases 1-5: Markov chain + Monte Carlo (~35s, every 6 hours)
  Phases 6-8: Pythia fine-tune + GPT-2 MC + blend (hours, auto-triggered when corpus grows)

Set-and-forget: the pipeline self-regulates when to fine-tune based on
corpus growth since the last fine-tune run. No manual intervention needed.
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

# How many new speeches (since last fine-tune) before we retrigger fine-tuning.
# This is higher than the Markov threshold because fine-tuning takes hours.
MIN_NEW_SPEECHES_FOR_FINE_TUNE = 50


class LocalPipeline:
    """Orchestrates the full local training pipeline.

    Automated lifecycle:
    1. Every 6h: scrape → Markov train → Monte Carlo → save predictions (35s)
    2. When corpus grows by 50+ speeches since last fine-tune AND no fine-tune
       is running: auto-launch Pythia fine-tune → GPT-2 MC → blend (hours)
    3. Fine-tuning runs in a background thread at lowest CPU priority.
       The bot continues trading on Markov predictions while it runs.
    """

    def __init__(self):
        self.trainer = MarkovChainTrainer(order=config.markov_order)
        self.colab_predictor = ColabPredictor()
        self.fine_tuner = None  # lazy-loaded
        self._lock = threading.Lock()
        self._ft_lock = threading.Lock()  # separate lock for fine-tuning
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
        # Track corpus size at time of last fine-tune
        self._last_fine_tune_corpus_size = self._get_last_fine_tune_corpus_size()
        # Pipeline-level fine-tune phase tracker (wraps the full Phase 6-8 lifecycle)
        self._ft_phase_status = {
            'state': 'idle',  # idle, phase6_training, phase7_monte_carlo, phase8_blending, complete, error
            'phase': None,    # 6, 7, or 8
            'stage': '',
            'progress': 0.0,  # 0-1 across all three phases
            'started_at': None,
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

        # Include fine-tune phase status (pipeline-level Phase 6-8 tracker)
        ft_phase = self._ft_phase_status.copy()
        ft = self._get_fine_tuner()
        if ft:
            ft_detail = ft.get_status()
            ft_phase['trainer'] = ft_detail  # nested fine-tuner detail

        if ft_phase['state'] != 'idle':
            status['fine_tune_status'] = ft_phase

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
        """Check if enough new data has accumulated to justify Markov retraining."""
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

    def should_fine_tune(self) -> bool:
        """Check if fine-tuning should auto-trigger.

        Returns True when ALL of:
        1. FINE_TUNE_ENABLED=true
        2. Fine-tuner dependencies are installed
        3. Not already running a fine-tune
        4. Corpus has grown by 50+ speeches since last fine-tune
           (or no fine-tuned model exists yet)
        """
        if not config.fine_tune_enabled:
            return False

        ft = self._get_fine_tuner()
        if not ft:
            return False

        # Don't start if already running
        if not self._ft_lock.acquire(blocking=False):
            return False
        self._ft_lock.release()

        ft_status = ft.get_status()
        if ft_status['state'] in ('training',):
            return False

        # Check corpus growth
        with get_session() as session:
            current_corpus = session.query(Speech).filter(
                Speech.transcript.isnot(None),
                Speech.is_processed == True,
                Speech.word_count >= 50,  # lower bar for fine-tuning (includes digests)
            ).count()

        new_since_ft = current_corpus - self._last_fine_tune_corpus_size

        if self._last_fine_tune_corpus_size == 0 and current_corpus > 0:
            # Never fine-tuned but have data — go
            logger.info(f"No previous fine-tune — triggering on {current_corpus} speeches")
            return True

        if new_since_ft >= MIN_NEW_SPEECHES_FOR_FINE_TUNE:
            logger.info(f"{new_since_ft} new speeches since last fine-tune — triggering")
            return True

        return False

    def run_full_pipeline(self) -> dict:
        """Run the complete local pipeline: Markov (always) + fine-tune (auto).

        Phases 1-5 run synchronously (~35s).
        Phases 6-8 launch in a background thread if should_fine_tune() is True.
        """
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

            # Phase 0: Social media corpus — initial import + digest refresh
            self._refresh_social_media()

            # Check if retraining is needed
            if not self.should_retrain():
                self._status.update(state='idle', stage='No retraining needed')
                self._log_event('Skipped — not enough new data')

                # Even if Markov doesn't retrain, check if fine-tuning should start
                if self.should_fine_tune():
                    self._launch_fine_tune_background()

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

            # Phase 6-8: Auto-trigger fine-tuning if corpus has grown enough
            if self.should_fine_tune():
                self._launch_fine_tune_background(term_list)

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

    # ── Fine-tune lifecycle ──

    def _launch_fine_tune_background(self, term_list: list[str] = None):
        """Launch the full fine-tune → MC → blend chain in a background thread.

        Acquires _ft_lock to prevent double-starts. The lock is held for the
        entire duration of fine-tuning (hours).
        """
        if not self._ft_lock.acquire(blocking=False):
            self._log_event('Fine-tune: skipped — already running')
            return

        # If no term_list provided, load from DB
        if not term_list:
            with get_session() as session:
                terms = session.query(Term).all()
                term_list = [t.normalized_term for t in terms]

        self._log_event('Phase 6-8: Auto-triggering fine-tune pipeline in background...')
        thread = threading.Thread(
            target=self._run_fine_tune_phases,
            args=(term_list,),
            daemon=True,
        )
        thread.start()

    def _run_fine_tune_phases(self, term_list: list[str]):
        """Phase 6-8: Fine-tune → Monte Carlo → Blend (runs in background thread).

        Holds _ft_lock for the entire duration. On completion, updates
        _last_fine_tune_corpus_size so the next trigger check works correctly.
        Updates _ft_phase_status throughout for the UI progress bar.
        """
        ft = self._get_fine_tuner()
        if not ft:
            self._log_event('Phase 6: Skipped — fine-tuner dependencies not installed')
            self._ft_lock.release()
            return

        try:
            self._ft_phase_status.update(
                state='phase6_training', phase=6,
                stage='Fine-tuning Pythia with LoRA',
                progress=0.0,
                started_at=datetime.utcnow().isoformat(),
                error=None,
            )

            # Phase 6: Fine-tune (~90% of total time)
            self._log_event('Phase 6: Fine-tuning Pythia with LoRA (this will take hours)...')
            ft_result = ft.train()

            if not ft_result:
                self._ft_phase_status.update(state='error', error='Fine-tuning returned None')
                self._log_event('Phase 6: Fine-tuning returned None — check logs')
                return
            if ft_result.get('status') in ('error', 'already_running'):
                self._ft_phase_status.update(state='error', error=ft_result.get('status'))
                self._log_event(f'Phase 6: Fine-tuning skipped — {ft_result.get("status")}')
                return
            if ft_result.get('status') == 'stopped':
                self._ft_phase_status.update(state='idle', stage='Stopped by user')
                self._log_event('Phase 6: Fine-tuning stopped by user — checkpoint saved')
                return

            self._ft_phase_status.update(progress=0.85)
            self._log_event(
                f'Phase 6: Fine-tuning complete — '
                f'{ft_result.get("total_steps", 0)} steps, '
                f'loss={ft_result.get("final_loss", "?")}, '
                f'{ft_result.get("training_seconds", 0):.0f}s'
            )

            # Phase 7: Pythia Monte Carlo (~10% of total time)
            if term_list:
                mc_sims = config.fine_tune_mc_sims
                self._ft_phase_status.update(
                    state='phase7_monte_carlo', phase=7,
                    stage=f'Pythia Monte Carlo ({mc_sims} sims)',
                    progress=0.85,
                )
                self._log_event(f'Phase 7: Running Pythia Monte Carlo ({mc_sims} sims)...')
                gpt2_predictions = ft.run_monte_carlo(term_list, num_simulations=mc_sims)

                if gpt2_predictions and gpt2_predictions.get('term_predictions'):
                    n_preds = len(gpt2_predictions['term_predictions'])
                    self._ft_phase_status.update(progress=0.95)
                    self._log_event(f'Phase 7: Pythia Monte Carlo complete — {n_preds} predictions')

                    # Phase 8: Blend with Markov predictions (~5%)
                    self._ft_phase_status.update(
                        state='phase8_blending', phase=8,
                        stage='Blending Markov + Pythia predictions',
                        progress=0.95,
                    )
                    self._blend_predictions(gpt2_predictions)
                    self._log_event('Phase 8: Blended Pythia + Markov predictions (60/40) and re-imported to DB')
                else:
                    self._log_event('Phase 7: Pythia Monte Carlo produced no predictions')

            # Update corpus tracking so we don't retrigger immediately
            with get_session() as session:
                self._last_fine_tune_corpus_size = session.query(Speech).filter(
                    Speech.transcript.isnot(None),
                    Speech.is_processed == True,
                    Speech.word_count >= 50,
                ).count()

            self._ft_phase_status.update(
                state='complete', phase=None,
                stage='Fine-tune pipeline complete',
                progress=1.0,
            )
            self._log_event(f'Fine-tune pipeline complete. Next trigger after {MIN_NEW_SPEECHES_FOR_FINE_TUNE}+ new speeches.')

            # Email notification
            try:
                from ..notifications.email_notifier import email_notifier
                if email_notifier.enabled:
                    email_notifier.send_critical_alert(
                        'Pythia Fine-Tune Complete',
                        f'Fine-tuned on {ft_result.get("corpus_size", "?")} speeches. '
                        f'Loss: {ft_result.get("final_loss", "?")}. '
                        f'Predictions blended and imported.',
                    )
            except Exception:
                pass

        except Exception as e:
            self._ft_phase_status.update(state='error', error=str(e))
            self._log_event(f'Fine-tune pipeline error: {e}')
            logger.error(f"Fine-tune phases failed: {e}")
        finally:
            self._ft_lock.release()

    def run_fine_tune_only(self):
        """Manually trigger the full fine-tune → MC → blend chain."""
        ft = self._get_fine_tuner()
        if not ft:
            return {'status': 'error', 'error': 'Fine-tuner dependencies not installed'}

        if not self._ft_lock.acquire(blocking=False):
            return {'status': 'already_running'}

        # Load terms
        with get_session() as session:
            terms = session.query(Term).all()
            term_list = [t.normalized_term for t in terms]

        self._log_event('Manual fine-tune triggered — running full Phase 6-8 chain')

        # Release the lock — _launch will re-acquire it
        self._ft_lock.release()

        self._launch_fine_tune_background(term_list)
        return {'status': 'started'}

    def _blend_predictions(self, gpt2_predictions: dict):
        """Blend Pythia Monte Carlo predictions with existing Markov predictions.

        Saves a blended predictions file with weighted average (60% Markov, 40% Pythia).
        Re-imports to DB so the ensemble predictor picks them up.
        """
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
                # Weighted blend: 60% Markov (more sims, proven), 40% Pythia
                blended_prob = 0.6 * mp['probability'] + 0.4 * gp['probability']
                entry = mp.copy()
                entry['probability'] = round(blended_prob, 4)
                entry['model_name'] = 'blended_markov_pythia'
                entry['markov_probability'] = mp['probability']
                entry['pythia_probability'] = gp['probability']
            else:
                entry = mp.copy()
            blended.append(entry)

        markov_data['term_predictions'] = blended
        markov_data['blended'] = True
        markov_data['blend_weights'] = {'markov': 0.6, 'pythia': 0.4}

        with open(pred_path, 'w') as f:
            json.dump(markov_data, f, indent=2)

        # Re-import to DB
        self.colab_predictor.save_to_database()

    # ── Social media ──

    def _refresh_social_media(self):
        """Phase 0: Ensure social media corpus is up to date.

        On first run: auto-imports the full Twitter archive (~56K tweets).
        Every run: scrapes latest Truth Social posts + rebuilds daily digests.
        """
        try:
            from ..scraper.social_media_importer import SocialMediaImporter
            importer = SocialMediaImporter()

            # Initial Twitter bulk import (one-time, if no tweets in DB)
            init_result = importer.ensure_initial_import()
            if init_result.get('status') == 'imported':
                self._log_event(
                    f'Phase 0: Initial Twitter import — '
                    f'{init_result.get("imported", 0)} tweets, '
                    f'{init_result.get("daily_digests_created", 0)} daily digests'
                )

            # Scrape latest Truth Social + rebuild recent digests
            new_posts = importer.scrape_latest_posts()
            if new_posts:
                self._log_event(f'Phase 0: Scraped {new_posts} new social media posts + rebuilt daily digests')

        except Exception as e:
            # Non-fatal — pipeline continues even if social media fails
            logger.warning(f"Social media refresh failed: {e}")
            self._log_event(f'Phase 0: Social media refresh failed (non-fatal): {e}')

    # ── Helpers ──

    def _get_last_training_count(self) -> int:
        """Get speech count from the last Markov training run."""
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

    def _get_last_fine_tune_corpus_size(self) -> int:
        """Get corpus size from the last fine-tune run."""
        try:
            with get_session() as session:
                latest = session.query(ModelVersion).filter_by(
                    model_type='gpt2_lora'
                ).order_by(ModelVersion.created_at.desc()).first()
                return latest.corpus_size if latest else 0
        except Exception:
            return 0

    def _notify_completion(self, version: str, train_result: dict,
                           pred_count: int):
        """Send email notification on successful Markov training."""
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
        """2C: Compute scenario weights based on the next known Trump event."""
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
