"""Automated pipeline: scrape → export → upload → trigger Colab → poll → import.

Orchestrates the full training cycle from data collection through model
fine-tuning to prediction import.  Designed to run as a scheduled job
or be triggered via the API.

Pipeline stages:
  1. Check if enough new speeches have accumulated since last training
  2. Export training corpus + context data
  3. Upload exports to Google Drive
  4. Write trigger file for Colab
  5. Poll Drive for completion signal
  6. Download predictions from Drive
  7. Import predictions into the trading database

Colab-side setup:
  The Colab notebook should be configured with Colab Pro's built-in
  scheduler to run periodically.  On startup it checks Drive for
  ``training_trigger.json``.  If found:
    • Downloads training data from Drive
    • Runs LoRA fine-tuning + Monte Carlo predictions
    • Writes ``predictions_latest.json`` to Drive/predictions
    • Writes ``training_complete.json`` to Drive/triggers
    • Deletes the trigger file
"""

import os
import time
import threading
from datetime import datetime
from typing import Optional
from loguru import logger

from .drive_sync import DriveSync
from .data_exporter import DataExporter
from .colab_integration import ColabPredictor
from ..database.db import get_session
from ..database.models import Speech


class ColabPipeline:
    """End-to-end automation of the scrape → train → predict cycle."""

    # Minimum new speeches required before triggering a retrain
    MIN_NEW_SPEECHES_FOR_RETRAIN = 5

    # Polling parameters for Colab completion
    POLL_TIMEOUT = 7200      # 2 hours (LoRA fine-tune on A100 ≈ 30–60 min)
    POLL_INTERVAL = 120      # check every 2 minutes

    def __init__(self):
        self.drive = DriveSync()
        self.exporter = DataExporter()
        self.colab = ColabPredictor()

        self._status: dict = {
            'state': 'idle',
            'last_run': None,
            'last_result': None,
            'last_training_speech_count': 0,
        }
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get current pipeline status."""
        with self._lock:
            status = dict(self._status)
        status['drive_configured'] = self.drive.is_configured
        return status

    def _set_state(self, state: str):
        with self._lock:
            self._status['state'] = state

    # ------------------------------------------------------------------
    # Retrain check
    # ------------------------------------------------------------------

    def should_retrain(self) -> bool:
        """Check if enough new data has accumulated to warrant retraining.

        Compares current speech count (with transcripts ≥ 100 words) to the
        count recorded at the last training run.
        """
        with get_session() as session:
            current_count = session.query(Speech).filter(
                Speech.transcript.isnot(None),
                Speech.word_count >= 100,
            ).count()

        last_count = self._status.get('last_training_speech_count', 0)
        new_speeches = current_count - last_count

        logger.debug(
            f"Retrain check: {current_count} total speeches, "
            f"{last_count} at last train, {new_speeches} new"
        )
        return new_speeches >= self.MIN_NEW_SPEECHES_FOR_RETRAIN

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self, force: bool = False) -> dict:
        """Execute the complete pipeline.

        Stages: export → upload → trigger → poll → import

        Args:
            force: Skip the ``should_retrain`` check and run anyway.

        Returns:
            Dict with results from each stage.
        """
        with self._lock:
            if self._status['state'] == 'running':
                return {'error': 'Pipeline already running', 'state': 'running'}
            self._status['state'] = 'running'

        results: dict = {
            'stages': {},
            'started_at': datetime.utcnow().isoformat(),
        }

        try:
            # Stage 0: check if retrain is warranted
            if not force and not self.should_retrain():
                results['skipped'] = True
                results['reason'] = 'Not enough new data for retraining'
                logger.info("[PIPELINE] Skipped — not enough new data since last training")
                return results

            # Stage 1: export training data
            stage_start = time.time()
            logger.info("=" * 60)
            logger.info("[PIPELINE] Stage 1/5: EXPORTING training data")
            logger.info("=" * 60)
            self._set_state('exporting')
            export_results = self._export_all()
            results['stages']['export'] = export_results
            corpus_info = export_results.get('corpus', {})
            logger.info(
                f"[PIPELINE] Export done in {time.time() - stage_start:.1f}s | "
                f"Speeches: {corpus_info.get('speech_count', '?')} | "
                f"Chunks: {corpus_info.get('chunk_count', '?')}"
            )
            if 'error' in corpus_info:
                results['error'] = 'Export failed: no training data'
                logger.error(f"[PIPELINE] FAILED at Stage 1: {corpus_info['error']}")
                return results

            # Stage 2: upload to Google Drive
            stage_start = time.time()
            logger.info("=" * 60)
            logger.info("[PIPELINE] Stage 2/5: UPLOADING to Google Drive")
            logger.info("=" * 60)
            self._set_state('uploading')
            upload_results = self.drive.upload_training_data()
            results['stages']['upload'] = upload_results
            uploaded_count = len(upload_results.get('uploaded', []))
            error_count = len(upload_results.get('errors', []))
            logger.info(
                f"[PIPELINE] Upload done in {time.time() - stage_start:.1f}s | "
                f"Files uploaded: {uploaded_count} | Errors: {error_count}"
            )
            if 'error' in upload_results:
                results['error'] = f"Upload failed: {upload_results['error']}"
                logger.error(f"[PIPELINE] FAILED at Stage 2: {upload_results['error']}")
                return results

            # Stage 3: trigger Colab training
            stage_start = time.time()
            logger.info("=" * 60)
            logger.info("[PIPELINE] Stage 3/5: TRIGGERING Colab training")
            logger.info("=" * 60)
            self._set_state('triggering')
            trigger_results = self.drive.write_trigger_file(
                trigger_type='full_pipeline',
                extra_data={
                    'corpus_stats': export_results.get('corpus', {}),
                    'triggered_by': 'auto_pipeline',
                },
            )
            results['stages']['trigger'] = trigger_results
            if 'error' in trigger_results:
                results['error'] = f"Trigger failed: {trigger_results['error']}"
                logger.error(f"[PIPELINE] FAILED at Stage 3: {trigger_results['error']}")
                return results
            logger.info(
                f"[PIPELINE] Trigger written in {time.time() - stage_start:.1f}s | "
                f"Now waiting for Colab to pick it up..."
            )
            logger.info(
                "[PIPELINE] Open your Colab notebook and run it, "
                "or wait for scheduled execution"
            )

            # Stage 4: poll for completion
            logger.info("=" * 60)
            logger.info("[PIPELINE] Stage 4/5: WAITING for Colab completion")
            logger.info(
                f"[PIPELINE] Polling every {self.POLL_INTERVAL}s, "
                f"timeout {self.POLL_TIMEOUT // 60}min"
            )
            logger.info("=" * 60)
            self._set_state('waiting_for_colab')
            completion = self._poll_for_completion()
            results['stages']['completion'] = completion or {'status': 'timeout'}
            if not completion:
                results['warning'] = (
                    'Colab training not yet complete. '
                    'Results will be imported on next poll cycle.'
                )
                logger.warning(
                    "[PIPELINE] Timed out waiting for Colab. "
                    "The scheduler will keep polling every 15 min."
                )
                return results
            logger.info("[PIPELINE] Colab training COMPLETE — importing results")

            # Stage 5: download and import predictions
            stage_start = time.time()
            logger.info("=" * 60)
            logger.info("[PIPELINE] Stage 5/5: IMPORTING predictions")
            logger.info("=" * 60)
            self._set_state('importing')
            import_results = self._import_predictions()
            results['stages']['import'] = import_results
            pred_count = import_results.get('download', {}).get('predictions_count', '?')
            logger.info(
                f"[PIPELINE] Import done in {time.time() - stage_start:.1f}s | "
                f"Predictions imported: {pred_count}"
            )

            # Record the speech count so we know when to retrain next
            self._record_speech_count()

            results['completed'] = True
            total_time = time.time() - results.get('_start_time', time.time())
            logger.info("=" * 60)
            logger.info("[PIPELINE] COMPLETE — all stages finished successfully")
            logger.info("=" * 60)

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Pipeline failed: {e}")

        finally:
            with self._lock:
                self._status['state'] = 'idle'
                self._status['last_run'] = datetime.utcnow().isoformat()
                self._status['last_result'] = results

        return results

    # ------------------------------------------------------------------
    # Partial operations (useful from the API)
    # ------------------------------------------------------------------

    def export_and_upload(self) -> dict:
        """Export training data and upload to Drive (no training trigger).

        Useful for syncing data before a manual Colab run.
        """
        results: dict = {}

        export_results = self._export_all()
        results['export'] = export_results

        if self.drive.is_configured:
            upload_results = self.drive.upload_training_data()
            results['upload'] = upload_results
        else:
            results['upload'] = {
                'skipped': True,
                'reason': 'Drive not configured',
            }

        return results

    def check_and_import(self) -> dict:
        """Check for completed Colab training and import results.

        Designed to be called periodically by the scheduler to pick up
        results from earlier-triggered Colab runs.
        """
        if not self.drive.is_configured:
            return {'skipped': True, 'reason': 'Drive not configured'}

        completion = self.drive.check_completion()
        if not completion:
            return {'status': 'no_completion_found'}

        logger.info("Found Colab completion signal, importing predictions")
        import_results = self._import_predictions()
        self._record_speech_count()

        return {
            'status': 'imported',
            'completion_data': completion,
            'import_results': import_results,
        }

    # ------------------------------------------------------------------
    # Background execution
    # ------------------------------------------------------------------

    def run_pipeline_async(self, force: bool = False):
        """Run the full pipeline in a background thread."""
        thread = threading.Thread(
            target=self.run_full_pipeline,
            args=(force,),
            daemon=True,
            name='colab-pipeline',
        )
        thread.start()
        logger.info("Pipeline started in background thread")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _export_all(self) -> dict:
        """Run all data exports (corpus + term context + event history)."""
        results: dict = {}
        for name, fn in [
            ('corpus', self.exporter.export_training_corpus),
            ('term_context', self.exporter.export_term_context),
            ('event_history', self.exporter.export_event_history),
        ]:
            try:
                results[name] = fn()
            except Exception as e:
                logger.error(f"Export {name} failed: {e}")
                results[name] = {'error': str(e)}
        return results

    def _poll_for_completion(self) -> Optional[dict]:
        """Poll Drive for the ``training_complete.json`` signal."""
        elapsed = 0
        poll_count = 0
        while elapsed < self.POLL_TIMEOUT:
            poll_count += 1
            completion = self.drive.check_completion()
            if completion:
                logger.info(
                    f"[PIPELINE] Colab completion detected after "
                    f"{elapsed // 60}m {elapsed % 60}s ({poll_count} polls)"
                )
                return completion

            remaining = (self.POLL_TIMEOUT - elapsed) // 60
            logger.info(
                f"[PIPELINE] Poll #{poll_count}: No completion yet | "
                f"Elapsed: {elapsed // 60}m {elapsed % 60}s | "
                f"Remaining: {remaining}m | "
                f"Next check in {self.POLL_INTERVAL}s"
            )
            time.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL

        logger.warning(
            f"[PIPELINE] Timed out after {self.POLL_TIMEOUT // 60}m "
            f"({poll_count} polls). Colab may still be running."
        )
        return None

    def _import_predictions(self) -> dict:
        """Download predictions from Drive and persist to the database."""
        results: dict = {}

        download = self.drive.download_predictions()
        results['download'] = download

        if not download.get('downloaded'):
            return results

        try:
            self.colab.save_to_database()
            results['saved_to_db'] = True
        except Exception as e:
            results['save_error'] = str(e)

        return results

    def _record_speech_count(self):
        """Record the current speech count for future retrain checks."""
        with get_session() as session:
            count = session.query(Speech).filter(
                Speech.transcript.isnot(None),
                Speech.word_count >= 100,
            ).count()
        with self._lock:
            self._status['last_training_speech_count'] = count
