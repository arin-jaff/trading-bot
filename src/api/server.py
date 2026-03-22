"""FastAPI backend server for the trading bot."""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from ..database.db import init_db, get_session
from ..database.models import Term, Market, TrumpEvent, Trade, TermPrediction, ModelVersion
from ..kalshi.client import KalshiClient
from ..kalshi.market_sync import MarketSync
from ..kalshi.trading_bot import TradingBot
from ..scraper.speech_scraper import SpeechScraper
from ..scraper.term_analyzer import TermAnalyzer
from ..scraper.event_tracker import EventTracker
from ..ml.predictor import TermPredictor
from ..ml.colab_integration import ColabPredictor
from ..scraper.live_monitor import LiveSpeechMonitor
from ..alerts import alert_manager
from ..config import config as app_config

app = FastAPI(title="Trump Mentions Trading Bot", version="1.0.0")

# --- In-memory job status tracker ---
import threading

_job_status: dict[str, dict] = {}
_job_lock = threading.Lock()


def _update_job(job_name: str, step: str, progress: int = 0,
                total: int = 0, done: bool = False, error: str = ''):
    with _job_lock:
        _job_status[job_name] = {
            'step': step,
            'progress': progress,
            'total': total,
            'done': done,
            'error': error,
            'updated_at': datetime.utcnow().isoformat(),
        }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
kalshi_client = KalshiClient()
market_sync = MarketSync(kalshi_client)
speech_scraper = SpeechScraper()
term_analyzer = TermAnalyzer()
event_tracker = EventTracker()
predictor = TermPredictor()
trading_bot = TradingBot(kalshi_client, predictor)
colab_predictor = ColabPredictor()
live_monitor = LiveSpeechMonitor()

# Pipeline: local or Colab depending on config
_pipeline = None
_drive_sync = None
if app_config.pipeline_mode == 'local':
    from ..ml.local_pipeline import LocalPipeline
    _pipeline = LocalPipeline()
else:
    from ..ml.colab_pipeline import ColabPipeline
    from ..ml.drive_sync import DriveSync
    _pipeline = ColabPipeline()
    _drive_sync = DriveSync()


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Database initialized")


# --- Market endpoints ---

@app.get("/api/markets")
def get_markets(status: Optional[str] = None):
    """Get all tracked markets.

    Returns markets sorted: active/open first (by close_time asc),
    then resolved/closed (by close_time desc — most recent first).

    Adds display_status field:
    - Active markets at 1c/99c: 'Virtually Certain' / 'Virtually Dead'
    - Resolved markets: 'Yes' / 'No' based on result
    - Otherwise: the raw status
    """
    with get_session() as session:
        query = session.query(Market)
        if status:
            query = query.filter(Market.status == status)
        all_markets = query.all()

        # Split into active and resolved, sort separately
        active = []
        resolved = []
        for m in all_markets:
            if m.status in ('active', 'open'):
                active.append(m)
            else:
                resolved.append(m)

        active.sort(key=lambda m: m.close_time or datetime.max)
        resolved.sort(key=lambda m: m.close_time or datetime.min, reverse=True)

        def _market_dict(m):
            yes_price = m.yes_price
            is_active = m.status in ('active', 'open')
            is_resolved = m.result in ('yes', 'no')

            # Determine display status
            if is_resolved:
                display_status = 'Yes' if m.result == 'yes' else 'No'
            elif is_active and yes_price is not None:
                if yes_price >= 0.99:
                    display_status = 'Virtually Certain'
                elif yes_price <= 0.01:
                    display_status = 'Virtually Dead'
                else:
                    display_status = m.status
            else:
                display_status = m.status

            return {
                'id': m.id,
                'ticker': m.kalshi_ticker,
                'title': m.title,
                'subtitle': m.subtitle,
                'status': m.status,
                'display_status': display_status,
                'yes_price': yes_price,
                'no_price': m.no_price,
                'volume': m.volume,
                'close_time': m.close_time.isoformat() if m.close_time else None,
                'expiration_time': m.expiration_time.isoformat() if m.expiration_time else None,
                'result': m.result,
                'terms': [t.term for t in m.terms],
            }

        return [_market_dict(m) for m in active + resolved]


@app.get("/api/markets/weekly-payouts")
def get_weekly_payouts(weeks: int = 12):
    """Get weekly payout data — which terms resolved yes/no each week."""
    from collections import defaultdict
    with get_session() as session:
        # Get all settled markets with results
        settled = session.query(Market).filter(
            Market.result.in_(['yes', 'no']),
            Market.close_time.isnot(None),
        ).all()

        # Group by week
        weekly: dict = defaultdict(lambda: {'yes': [], 'no': []})
        for m in settled:
            week_start = m.close_time - timedelta(days=m.close_time.weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            terms = [t.term for t in m.terms]
            weekly[week_key][m.result].append({
                'ticker': m.kalshi_ticker,
                'title': m.title,
                'terms': terms,
                'close_time': m.close_time.isoformat(),
                'yes_price': m.yes_price,
                'volume': m.volume,
            })

        # Sort by week descending, limit to N weeks
        sorted_weeks = sorted(weekly.items(), key=lambda x: x[0], reverse=True)[:weeks]

        return [
            {
                'week': week,
                'yes_count': len(data['yes']),
                'no_count': len(data['no']),
                'yes_markets': data['yes'],
                'no_markets': data['no'],
                'yes_terms': list(set(t for m in data['yes'] for t in m['terms'])),
                'no_terms': list(set(t for m in data['no'] for t in m['terms'])),
            }
            for week, data in sorted_weeks
        ]


@app.post("/api/markets/sync")
def sync_markets(background_tasks: BackgroundTasks):
    """Trigger market sync from Kalshi."""
    background_tasks.add_task(_run_market_sync)
    return {"status": "sync started"}


def _run_market_sync():
    try:
        _update_job('market_sync', 'Fetching markets from Kalshi...')
        stats = market_sync.sync_markets()
        _update_job('market_sync', f'Done: {stats}', done=True)
        logger.info(f"Market sync: {stats}")
    except Exception as e:
        _update_job('market_sync', '', done=True, error=str(e))
        logger.error(f"Market sync failed: {e}")


# --- Terms endpoints ---

@app.get("/api/terms")
def get_terms():
    """Get all tracked terms with stats."""
    return market_sync.get_all_terms()


@app.get("/api/terms/{term_id}/history")
def get_term_history(term_id: int, days: int = 365):
    """Get term usage time series."""
    return term_analyzer.get_term_time_series(term_id, days)


@app.get("/api/terms/report")
def get_term_report():
    """Get full term frequency report."""
    return term_analyzer.get_term_frequency_report()


# --- Speech endpoints ---

@app.post("/api/speeches/scrape")
def scrape_speeches(background_tasks: BackgroundTasks):
    """Trigger speech scraping from all sources."""
    background_tasks.add_task(_run_speech_scrape)
    return {"status": "scraping started"}


def _run_speech_scrape():
    try:
        _update_job('speech_scrape', 'Scraping speeches from 10 sources...')
        stats = speech_scraper.scrape_all_sources()
        _update_job('speech_scrape', f'Scraped {stats}. Analyzing terms...')
        processed = term_analyzer.process_all_unprocessed()
        _update_job('speech_scrape', f'Done: {stats}, processed {processed} speeches', done=True)
        logger.info(f"Speech scrape: {stats}, processed {processed}")
    except Exception as e:
        _update_job('speech_scrape', '', done=True, error=str(e))
        logger.error(f"Speech scrape failed: {e}")


@app.get("/api/speeches/stats")
def get_speech_stats():
    """Get speech collection statistics."""
    from ..database.models import Speech
    with get_session() as session:
        total = session.query(Speech).count()
        with_transcript = session.query(Speech).filter(
            Speech.transcript.isnot(None)
        ).count()
        processed = session.query(Speech).filter_by(is_processed=True).count()
        return {
            'total_speeches': total,
            'with_transcripts': with_transcript,
            'processed': processed,
        }


# --- Event endpoints ---

@app.get("/api/events")
def get_events(days: int = 30):
    """Get upcoming Trump events."""
    return event_tracker.get_upcoming_events(days)


@app.get("/api/events/live")
def get_live_events():
    """Get currently live events."""
    event_tracker.check_and_update_live_status()
    return event_tracker.get_live_events()


@app.post("/api/events/update")
def update_events(background_tasks: BackgroundTasks):
    """Trigger event discovery."""
    background_tasks.add_task(event_tracker.update_events)
    return {"status": "event update started"}


# --- Predictions endpoints ---

@app.get("/api/predictions")
def get_predictions():
    """Get latest predictions for all terms."""
    return predictor.predict_all_terms()


@app.post("/api/predictions/generate")
def generate_predictions(background_tasks: BackgroundTasks):
    """Trigger new prediction generation."""
    background_tasks.add_task(_run_predictions)
    return {"status": "prediction generation started"}


def _run_predictions():
    try:
        _update_job('predictions', 'Loading Monte Carlo predictions...')
        preds = predictor.predict_all_terms()
        _update_job('predictions', f'Saving {len(preds)} predictions...')
        predictor.save_predictions(preds)
        _update_job('predictions', f'Done: {len(preds)} predictions generated', done=True)
        logger.info(f"Generated {len(preds)} predictions")
    except Exception as e:
        _update_job('predictions', '', done=True, error=str(e))
        logger.error(f"Prediction generation failed: {e}")


# --- Trading endpoints ---

@app.get("/api/trading/suggestions")
def get_suggestions():
    """Get trading suggestions."""
    return trading_bot.generate_suggestions()


@app.get("/api/trading/portfolio")
def get_portfolio():
    """Get portfolio summary."""
    return trading_bot.get_portfolio_summary()


@app.get("/api/trading/config")
def get_bot_config():
    """Get trading bot configuration."""
    return trading_bot.get_config()


class BotConfigUpdate(BaseModel):
    max_position_size: Optional[int] = None
    max_daily_loss: Optional[float] = None
    max_total_exposure: Optional[float] = None
    min_edge_threshold: Optional[float] = None
    min_confidence: Optional[float] = None
    auto_trade: Optional[bool] = None
    paper_mode: Optional[bool] = None
    kelly_fraction: Optional[float] = None


@app.put("/api/trading/config")
def update_bot_config(config: BotConfigUpdate):
    """Update trading bot configuration."""
    updates = {k: v for k, v in config.dict().items() if v is not None}
    trading_bot.update_config(**updates)
    return trading_bot.get_config()


class TradeRequest(BaseModel):
    market_ticker: str
    side: str
    quantity: int
    price_cents: Optional[int] = None


@app.post("/api/trading/execute")
def execute_trade(req: TradeRequest):
    """Execute a trade."""
    suggestion = {
        'market_ticker': req.market_ticker,
        'suggested_side': req.side,
        'suggested_quantity': req.quantity,
        'market_yes_price': (req.price_cents or 50) / 100,
        'edge': 0,
        'reasoning': 'manual trade',
    }
    result = trading_bot.execute_trade(suggestion, req.quantity, require_confirmation=False)
    if result and result.get('status') == 'error':
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result


@app.post("/api/kalshi/login")
def kalshi_login():
    """Login to Kalshi."""
    success = kalshi_client.login()
    if success:
        return {"status": "logged in", "member_id": kalshi_client.member_id}
    raise HTTPException(status_code=401, detail="Login failed")


# --- System endpoints ---

@app.post("/api/system/full-refresh")
def full_refresh(background_tasks: BackgroundTasks):
    """Run full data refresh: sync markets, scrape speeches, update events, generate predictions."""
    background_tasks.add_task(_full_refresh)
    return {"status": "full refresh started"}


def _full_refresh():
    try:
        _update_job('full_refresh', 'Step 1/6: Logging in to Kalshi...')
        kalshi_client.login()
        _update_job('full_refresh', 'Step 2/6: Syncing markets...', progress=1, total=6)
        market_sync.sync_markets()
        _update_job('full_refresh', 'Step 3/6: Scraping speeches...', progress=2, total=6)
        speech_scraper.scrape_all_sources()
        _update_job('full_refresh', 'Step 4/6: Analyzing terms...', progress=3, total=6)
        term_analyzer.process_all_unprocessed()
        _update_job('full_refresh', 'Step 5/6: Updating events...', progress=4, total=6)
        event_tracker.update_events()
        _update_job('full_refresh', 'Step 6/6: Generating predictions...', progress=5, total=6)
        preds = predictor.predict_all_terms()
        predictor.save_predictions(preds)
        _update_job('full_refresh', f'Done: {len(preds)} predictions', progress=6, total=6, done=True)
        logger.info("Full refresh complete")
    except Exception as e:
        _update_job('full_refresh', '', done=True, error=str(e))
        logger.error(f"Full refresh failed: {e}")


# --- Live monitoring endpoints ---

@app.post("/api/live/start")
def start_live_monitoring():
    """Start live speech monitoring."""
    live_monitor.start_monitoring()
    return {"status": "monitoring started"}


@app.post("/api/live/stop")
def stop_live_monitoring():
    """Stop live speech monitoring."""
    live_monitor.stop_monitoring()
    return {"status": "monitoring stopped"}


@app.get("/api/live/status")
def get_live_status():
    """Get live monitoring status."""
    return live_monitor.get_live_status()


# --- ML Model endpoints ---

@app.post("/api/ml/train")
def train_models():
    """Deprecated — use POST /api/pipeline/run instead."""
    return {"status": "deprecated", "message": "Use POST /api/pipeline/run"}


@app.get("/api/ml/info")
def get_model_info():
    """Deprecated — use GET /api/model/status instead."""
    return {"status": "deprecated", "message": "Use GET /api/model/status"}


@app.get("/api/ml/predictions")
def get_ml_predictions():
    """Deprecated — use GET /api/predictions/final instead."""
    return {"status": "deprecated", "message": "Use GET /api/predictions/final"}


@app.get("/api/colab/predictions")
def get_colab_predictions():
    """Get predictions from Colab fine-tuned model."""
    return colab_predictor.get_predictions()


@app.post("/api/colab/import")
def import_colab_predictions(file_path: str):
    """Import a predictions JSON from Colab."""
    return colab_predictor.import_predictions_file(file_path)


@app.post("/api/colab/save-to-db")
def save_colab_to_db():
    """Save Colab predictions to the trading database."""
    colab_predictor.save_to_database()
    return {"status": "saved"}


@app.get("/api/colab/discovered-phrases")
def get_discovered_phrases():
    """Get new phrases discovered by Monte Carlo that aren't in Kalshi's term list."""
    return colab_predictor.get_discovered_phrases()


# --- Pipeline endpoints ---

@app.get("/api/pipeline/status")
def get_pipeline_status():
    """Get the status of the automated training pipeline."""
    status = {'pipeline': _pipeline.get_status(), 'mode': app_config.pipeline_mode}
    if _drive_sync:
        status['drive'] = _drive_sync.get_status()
    return status


@app.get("/api/pipeline/training-status")
def get_training_status():
    """Get real-time training progress for GUI polling."""
    return _pipeline.get_status()


@app.get("/api/pipeline/log")
def get_pipeline_log(limit: int = 50):
    """Get recent pipeline log entries."""
    return _pipeline.get_log(limit)


@app.post("/api/pipeline/run")
def run_pipeline(background_tasks: BackgroundTasks):
    """Trigger the training pipeline (local or Colab depending on config)."""
    _pipeline.run_pipeline_async()
    return {"status": "pipeline started", "mode": app_config.pipeline_mode}


@app.post("/api/pipeline/export-upload")
def pipeline_export_upload(background_tasks: BackgroundTasks):
    """Export training data and upload to Google Drive (Colab mode only)."""
    if app_config.pipeline_mode == 'local':
        return {"status": "disabled", "message": "Set PIPELINE_MODE=colab to enable"}
    background_tasks.add_task(_run_export_upload)
    return {"status": "export and upload started"}


def _run_export_upload():
    try:
        _update_job('export_upload', 'Exporting training data...')
        result = _pipeline.export_and_upload()
        _update_job('export_upload', f'Done: {result}', done=True)
    except Exception as e:
        _update_job('export_upload', '', done=True, error=str(e))
        logger.error(f"Export & upload failed: {e}")


@app.post("/api/pipeline/trigger-training")
def trigger_training():
    """Upload latest data to Drive and write a trigger file for Colab."""
    if not _drive_sync:
        return {"status": "disabled", "message": "Set PIPELINE_MODE=colab to enable"}
    upload = _drive_sync.upload_training_data()
    if 'error' in upload:
        raise HTTPException(status_code=500, detail=upload['error'])
    trigger = _drive_sync.write_trigger_file(
        trigger_type='manual',
        extra_data={'triggered_from': 'api'},
    )
    return {'upload': upload, 'trigger': trigger}


@app.post("/api/pipeline/poll")
def poll_colab_results():
    """Check if Colab training has completed and import results."""
    if app_config.pipeline_mode == 'local':
        return {"status": "disabled", "message": "Set PIPELINE_MODE=colab to enable"}
    return _pipeline.check_and_import()


@app.get("/api/drive/status")
def get_drive_status():
    """Get Google Drive integration status."""
    if not _drive_sync:
        return {"status": "disabled", "message": "Set PIPELINE_MODE=colab to enable"}
    return _drive_sync.get_status()


@app.post("/api/drive/upload")
def upload_to_drive():
    """Upload training exports to Google Drive."""
    if not _drive_sync:
        return {"status": "disabled", "message": "Set PIPELINE_MODE=colab to enable"}
    return _drive_sync.upload_training_data()


@app.post("/api/drive/download-predictions")
def download_predictions_from_drive():
    """Download latest predictions from Google Drive."""
    if not _drive_sync:
        return {"status": "disabled", "message": "Set PIPELINE_MODE=colab to enable"}
    return _drive_sync.download_predictions()


# --- Job status endpoint ---

@app.get("/api/jobs/status")
def get_job_statuses():
    """Get status of all background jobs."""
    with _job_lock:
        return dict(_job_status)


@app.get("/api/jobs/status/{job_name}")
def get_job_status(job_name: str):
    """Get status of a specific background job."""
    with _job_lock:
        return _job_status.get(job_name, {'step': 'idle', 'done': True})


@app.post("/api/drive/download-and-import")
def download_and_import_predictions(background_tasks: BackgroundTasks):
    """Download predictions from Drive and import into DB."""
    if not _drive_sync:
        return {"status": "disabled", "message": "Set PIPELINE_MODE=colab to enable"}
    background_tasks.add_task(_run_download_import)
    return {"status": "downloading predictions from Drive..."}


def _run_download_import():
    try:
        _update_job('drive_download', 'Downloading predictions_latest.json from Drive...')
        result = _drive_sync.download_predictions()
        if 'error' in result:
            _update_job('drive_download', '', done=True, error=result['error'])
            return
        _update_job('drive_download', 'Saving to database...')
        colab_predictor.save_to_database()
        _update_job('drive_download', 'Done: predictions imported', done=True)
        logger.info(f"Drive download & import: {result}")
    except Exception as e:
        _update_job('drive_download', '', done=True, error=str(e))
        logger.error(f"Drive download failed: {e}")


# --- Alert endpoints ---

@app.get("/api/alerts")
def get_alerts(limit: int = 50, alert_type: Optional[str] = None,
               unread_only: bool = False):
    """Get recent alerts."""
    return alert_manager.get_recent_alerts(limit, alert_type, unread_only)


@app.get("/api/alerts/count")
def get_unread_count():
    """Get unread alert count."""
    return {"unread": alert_manager.get_unread_count()}


@app.post("/api/alerts/{alert_id}/read")
def mark_alert_read(alert_id: int):
    """Mark an alert as read."""
    alert_manager.mark_read(alert_id)
    return {"status": "ok"}


# --- Config endpoints ---

@app.get("/api/config/status")
def get_config_status():
    """Get configuration status."""
    return app_config.get_status()


# --- System endpoints ---

@app.get("/api/model/status")
def get_model_status():
    """Get TrumpGPT model status: weights, terms, scenario info, last run."""
    # Load latest Colab predictions file for metadata
    import glob as _glob
    pred_dir = os.path.join('data', 'predictions')
    latest_path = os.path.join(pred_dir, 'predictions_latest.json')
    colab_meta = {}
    term_predictions = []

    if os.path.exists(latest_path):
        with open(latest_path) as f:
            colab_meta = json.load(f)
        term_predictions = colab_meta.get('term_predictions', [])

    # Ensemble weights from the local predictor
    ensemble_weights = predictor.model_weights

    # Count terms in DB
    with get_session() as session:
        total_terms = session.query(Term).count()
        total_predictions = session.query(TermPrediction).count()

    # What a new iteration would bring
    from ..database.models import Speech
    with get_session() as session:
        new_speeches = session.query(Speech).filter(
            Speech.is_processed == True,
            Speech.created_at >= datetime.utcnow() - timedelta(days=1),
        ).count()

    # Get active model version
    with get_session() as session:
        active_model = session.query(ModelVersion).filter_by(
            is_active=True
        ).order_by(ModelVersion.created_at.desc()).first()
        model_version_info = None
        if active_model:
            model_version_info = {
                'version': active_model.version,
                'model_type': active_model.model_type,
                'markov_order': active_model.markov_order,
                'corpus_size': active_model.corpus_size,
                'training_duration': active_model.training_duration_seconds,
                'trained_at': active_model.created_at.isoformat() if active_model.created_at else None,
            }

    sim_params = colab_meta.get('simulation_params', {})
    model_type = sim_params.get('model_type', 'markov_chain' if app_config.pipeline_mode == 'local' else 'colab_llm')
    method = 'Markov Chain + Monte Carlo' if model_type == 'markov_chain' else 'LoRA fine-tune + Monte Carlo'

    return {
        'model_name': 'TrumpGPT',
        'version': model_version_info.get('version') if model_version_info else None,
        'version_info': model_version_info,
        'pipeline_mode': app_config.pipeline_mode,
        'base_model': sim_params.get('base_model', model_type),
        'method': method,
        'ensemble_weights': ensemble_weights,
        'last_run': colab_meta.get('generated_at'),
        'simulation_params': colab_meta.get('simulation_params', {}),
        'scenario_weights': colab_meta.get('scenario_weights_used', {}),
        'scenario_counts': colab_meta.get('scenario_counts', {}),
        'gemini_enrichment': colab_meta.get('gemini_enrichment', {}),
        'total_terms_tracked': total_terms,
        'total_predictions_in_db': total_predictions,
        'colab_predictions_count': len(term_predictions),
        'discovered_phrases_count': len(colab_meta.get('discovered_phrases', [])),
        'new_iteration_info': {
            'new_speeches_last_24h': new_speeches,
            'would_retrain': new_speeches >= 5,
            'description': f'{new_speeches} new speeches scraped in last 24h. '
                           f'{"Ready for retraining." if new_speeches >= 5 else "Need " + str(5 - new_speeches) + " more for retrain trigger."}',
        },
        'top_predictions': [
            {'term': p['term'], 'probability': p['probability'],
             'recency_weight': p.get('recency_weight', 1.0)}
            for p in term_predictions[:15]
        ],
    }


@app.get("/api/predictions/final")
def get_final_predictions():
    """Get final combined predictions for upcoming markets.

    Blends:
    - Historical market results (what he said vs didn't)
    - TrumpGPT Monte Carlo predictions
    - Local ensemble predictor
    """
    from collections import defaultdict
    from ..database.models import Speech, TermOccurrence

    with get_session() as session:
        # Get active markets
        active_markets = session.query(Market).filter(
            Market.status.in_(['active', 'open'])
        ).order_by(Market.close_time).all()

        if not active_markets:
            return []

        # Historical data: for each term, how often has he said it?
        total_processed = session.query(Speech).filter_by(is_processed=True).count()

        # Get latest ensemble predictions
        ensemble_preds = {}
        try:
            all_preds = predictor.predict_all_terms()
            ensemble_preds = {p['term'].lower().strip(): p for p in all_preds}
        except Exception as e:
            logger.warning(f"Ensemble predictions failed: {e}")

        # Load Colab predictions
        colab_preds = {}
        try:
            colab = colab_predictor.get_predictions()
            colab_preds = {p['term'].lower().strip(): p for p in colab}
        except Exception:
            pass

        # Historical settled results for context
        settled = session.query(Market).filter(
            Market.result.in_(['yes', 'no'])
        ).all()
        historical_results = defaultdict(lambda: {'yes': 0, 'no': 0})
        for m in settled:
            for t in m.terms:
                historical_results[t.normalized_term][m.result] += 1

        results = []
        for market in active_markets:
            for term in market.terms:
                norm = term.normalized_term.lower().strip()

                # Historical stats
                occ_count = session.query(TermOccurrence).filter_by(
                    term_id=term.id
                ).count()
                speeches_with_term = session.query(
                    TermOccurrence.speech_id
                ).filter_by(term_id=term.id).distinct().count()
                hist_rate = speeches_with_term / max(1, total_processed)

                # Past market outcomes for this term
                hist = historical_results.get(norm, {'yes': 0, 'no': 0})
                past_yes = hist['yes']
                past_no = hist['no']
                past_total = past_yes + past_no
                historical_market_rate = past_yes / max(1, past_total) if past_total > 0 else None

                # Ensemble prediction
                ens = ensemble_preds.get(norm, {})
                ensemble_prob = ens.get('probability')
                component_scores = ens.get('component_scores', {})

                # Colab prediction
                col = colab_preds.get(norm, {})
                colab_prob = col.get('probability')
                recency_weight = col.get('recency_weight', 1.0)
                by_scenario = col.get('by_scenario', {})

                # Final blended probability
                signals = []
                if ensemble_prob is not None:
                    signals.append(('ensemble', ensemble_prob, 0.5))
                if historical_market_rate is not None:
                    signals.append(('historical_markets', historical_market_rate, 0.2))
                if hist_rate > 0:
                    signals.append(('speech_frequency', min(1.0, hist_rate * 3), 0.3))

                if signals:
                    total_w = sum(s[2] for s in signals)
                    final_prob = sum(s[1] * s[2] / total_w for s in signals)
                else:
                    final_prob = market.yes_price or 0.5

                results.append({
                    'market_ticker': market.kalshi_ticker,
                    'market_title': market.title,
                    'term': term.term,
                    'close_time': market.close_time.isoformat() if market.close_time else None,
                    'market_yes_price': market.yes_price,
                    'market_no_price': market.no_price,
                    'final_probability': round(final_prob, 4),
                    'edge': round(final_prob - (market.yes_price or 0.5), 4),
                    'ensemble_probability': ensemble_prob,
                    'colab_probability': colab_prob,
                    'recency_weight': recency_weight,
                    'historical_speech_rate': round(hist_rate, 4),
                    'historical_market_record': {
                        'yes': past_yes, 'no': past_no,
                        'rate': round(historical_market_rate, 4) if historical_market_rate else None,
                    },
                    'speeches_with_term': speeches_with_term,
                    'total_occurrences': occ_count,
                    'component_scores': component_scores,
                    'by_scenario': by_scenario,
                })

        results.sort(key=lambda x: abs(x['edge']), reverse=True)
        return results


@app.get("/api/system/hardware")
def get_hardware_status():
    """Get Raspberry Pi / system hardware metrics."""
    import psutil
    import platform

    # CPU, RAM, disk
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    load_1, load_5, load_15 = psutil.getloadavg()
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime_hours = (datetime.utcnow() - boot_time).total_seconds() / 3600

    # Temperature (Pi-specific via vcgencmd, fallback for other platforms)
    temperature = None
    try:
        import subprocess
        result = subprocess.run(
            ['vcgencmd', 'measure_temp'], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            temp_str = result.stdout.strip()  # "temp=52.0'C"
            temperature = float(temp_str.split('=')[1].split("'")[0])
    except Exception:
        # Try psutil sensors as fallback
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        temperature = entries[0].current
                        break
        except Exception:
            pass

    return {
        'cpu_percent': cpu_percent,
        'ram_total_gb': round(ram.total / (1024**3), 2),
        'ram_used_gb': round(ram.used / (1024**3), 2),
        'ram_percent': ram.percent,
        'disk_total_gb': round(disk.total / (1024**3), 2),
        'disk_used_gb': round(disk.used / (1024**3), 2),
        'disk_percent': round(disk.percent, 1),
        'temperature_c': temperature,
        'uptime_hours': round(uptime_hours, 1),
        'load_avg_1m': round(load_1, 2),
        'load_avg_5m': round(load_5, 2),
        'load_avg_15m': round(load_15, 2),
        'platform': platform.machine(),
        'python_version': platform.python_version(),
    }


@app.get("/api/trades/history")
def get_trade_history(page: int = 1, per_page: int = 50,
                      status: Optional[str] = None):
    """Get paginated trade history with P&L."""
    with get_session() as session:
        query = session.query(Trade).join(Market)
        if status and status != 'all':
            query = query.filter(Trade.status == status)

        total = query.count()
        trades = query.order_by(Trade.created_at.desc()).offset(
            (page - 1) * per_page
        ).limit(per_page).all()

        # Summary stats
        all_trades = session.query(Trade).all()
        filled = [t for t in all_trades if t.pnl is not None]
        wins = [t for t in filled if (t.pnl or 0) > 0]
        total_pnl = sum(t.pnl or 0 for t in filled)

        return {
            'trades': [
                {
                    'id': t.id,
                    'market_ticker': t.market.kalshi_ticker if t.market else '?',
                    'market_title': t.market.title if t.market else '?',
                    'side': t.side,
                    'action': t.action,
                    'quantity': t.quantity,
                    'price': t.price,
                    'fill_price': t.fill_price,
                    'pnl': t.pnl,
                    'status': t.status,
                    'strategy': t.strategy,
                    'reasoning': t.reasoning,
                    'created_at': t.created_at.isoformat() if t.created_at else None,
                }
                for t in trades
            ],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page,
            },
            'summary': {
                'total_trades': len(all_trades),
                'filled_trades': len(filled),
                'win_count': len(wins),
                'win_rate': round(len(wins) / max(1, len(filled)), 4),
                'total_pnl': round(total_pnl, 2),
                'avg_trade_size': round(
                    sum(t.quantity for t in all_trades) / max(1, len(all_trades)), 1
                ),
            },
        }


def _get_version_accuracy(session, model_version_id: int) -> Optional[dict]:
    """Compute accuracy for a specific model version from tagged predictions."""
    from ..database.models import TermPrediction
    preds = session.query(TermPrediction).filter(
        TermPrediction.model_version_id == model_version_id,
        TermPrediction.was_correct.isnot(None),
    ).all()
    if not preds:
        return None
    correct = sum(1 for p in preds if p.was_correct)
    brier = sum((p.probability - (1.0 if p.was_correct else 0.0)) ** 2 for p in preds) / len(preds)
    return {
        'hit_rate': round(correct / len(preds), 4),
        'brier_score': round(brier, 4),
        'evaluated': len(preds),
    }


@app.get("/api/model/versions")
def get_model_versions():
    """Get all model version records."""
    with get_session() as session:
        versions = session.query(ModelVersion).order_by(
            ModelVersion.created_at.desc()
        ).all()

        return [
            {
                'id': v.id,
                'version': v.version,
                'model_type': v.model_type,
                'markov_order': v.markov_order,
                'corpus_size': v.corpus_size,
                'corpus_word_count': v.corpus_word_count,
                'training_duration_seconds': v.training_duration_seconds,
                'simulation_count': v.simulation_count,
                'prediction_count': v.prediction_count,
                'metrics': v.metrics,
                'is_active': v.is_active,
                'notes': v.notes,
                'accuracy': _get_version_accuracy(session, v.id),
                'created_at': v.created_at.isoformat() if v.created_at else None,
            }
            for v in versions
        ]


# --- Social Media Import endpoints ---

_social_importer = None


def _get_social_importer():
    global _social_importer
    if _social_importer is None:
        from ..scraper.social_media_importer import SocialMediaImporter
        _social_importer = SocialMediaImporter()
    return _social_importer


@app.post("/api/social-media/import-twitter")
def import_twitter(background_tasks: BackgroundTasks):
    """Download and import Trump Twitter archive."""
    importer = _get_social_importer()
    background_tasks.add_task(_run_twitter_import, importer)
    return {"status": "Twitter import started"}


def _run_twitter_import(importer):
    try:
        importer.import_twitter_archive()
    except Exception as e:
        logger.error(f"Twitter import failed: {e}")


class TruthImportRequest(BaseModel):
    file_path: str


@app.post("/api/social-media/import-truth")
def import_truth_social(req: TruthImportRequest, background_tasks: BackgroundTasks):
    """Import Truth Social posts from a JSON dump."""
    importer = _get_social_importer()
    background_tasks.add_task(_run_truth_import, importer, req.file_path)
    return {"status": "Truth Social import started"}


def _run_truth_import(importer, file_path: str):
    try:
        importer.import_truth_social(file_path)
    except Exception as e:
        logger.error(f"Truth Social import failed: {e}")


@app.post("/api/social-media/scrape-truth")
def scrape_truth_social(background_tasks: BackgroundTasks):
    """Scrape latest Truth Social posts via RSS/API (no file needed)."""
    importer = _get_social_importer()
    background_tasks.add_task(_run_truth_scrape, importer)
    return {"status": "Truth Social scrape started"}


def _run_truth_scrape(importer):
    try:
        new_posts = importer.scrape_latest_posts()
        logger.info(f"Truth Social scrape: {new_posts} new posts")
    except Exception as e:
        logger.error(f"Truth Social scrape failed: {e}")


@app.get("/api/social-media/import-status")
def get_import_status():
    """Get social media import progress."""
    importer = _get_social_importer()
    return importer.get_status()


@app.get("/api/social-media/stats")
def get_social_media_stats():
    """Get social media corpus statistics."""
    importer = _get_social_importer()
    return importer.get_stats()


@app.get("/api/social-media/recent-posts")
def get_recent_social_posts(limit: int = 5):
    """Get the most recent social media posts from the DB."""
    from ..database.models import Speech

    with get_session() as session:
        posts = session.query(Speech).filter(
            Speech.speech_type == 'social_media',
            Speech.transcript.isnot(None),
        ).order_by(Speech.date.desc()).limit(limit).all()

        return [
            {
                'source': p.source,
                'text': p.transcript[:280] if p.transcript else '',
                'date': p.date.isoformat() if p.date else None,
                'word_count': p.word_count,
            }
            for p in posts
        ]


# --- Fine-tuning + Pythia predictions ---

@app.get("/api/fine-tune/pythia-status")
def get_pythia_status():
    """Check if Pythia predictions are available on disk."""
    pythia_path = os.path.join('data', 'predictions', 'predictions_pythia.json')
    if os.path.exists(pythia_path):
        mtime = os.path.getmtime(pythia_path)
        with open(pythia_path) as f:
            data = json.load(f)
        return {
            "available": True,
            "predictions": len(data.get('term_predictions', [])),
            "generated_at": data.get('generated_at'),
            "last_synced": datetime.fromtimestamp(mtime).isoformat(),
            "model": data.get('simulation_params', {}).get('model_name', 'pythia'),
        }
    return {"available": False}


@app.get("/api/fine-tune/pi-status")
def get_pi_fine_tune_status():
    """Get Pi-native fine-tuning status and configuration."""
    result = {
        "enabled": app_config.fine_tune_enabled,
        "model": app_config.fine_tune_model,
        "lora_rank": app_config.fine_tune_lora_rank,
        "epochs": app_config.fine_tune_epochs,
        "gradient_checkpointing": app_config.fine_tune_gradient_checkpointing,
        "schedule_hour": app_config.fine_tune_hour,
        "pytorch_available": False,
        "torch_version": None,
        "training_status": None,
        "loss_history": [],
    }

    # Check PyTorch availability
    try:
        import torch
        result["pytorch_available"] = True
        result["torch_version"] = torch.__version__
    except ImportError:
        pass

    # Check fine-tuner status if available
    try:
        from ..ml.fine_tuner import GPT2FineTuner
        fine_tuner = GPT2FineTuner()
        result["training_status"] = fine_tuner.get_status()
        result["loss_history"] = fine_tuner.get_loss_history()[-100:]
        result["has_trained_model"] = fine_tuner.has_trained_model()
    except Exception as e:
        result["error"] = str(e)

    return result


@app.post("/api/fine-tune/start")
def start_pi_fine_tuning(background_tasks: BackgroundTasks):
    """Manually trigger Pi fine-tuning."""
    if not app_config.fine_tune_enabled:
        raise HTTPException(status_code=400, detail="Fine-tuning disabled")

    try:
        from ..ml.local_pipeline import LocalPipeline
        pipeline = LocalPipeline()
        background_tasks.add_task(pipeline.run_fine_tuning)
        return {"status": "Fine-tuning started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fine-tune/stop")
def stop_pi_fine_tuning():
    """Stop running Pi fine-tuning."""
    try:
        from ..ml.fine_tuner import GPT2FineTuner
        fine_tuner = GPT2FineTuner()
        fine_tuner.stop_training()
        return {"status": "Stop requested"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/full")
def run_full_pipeline(background_tasks: BackgroundTasks):
    """Run fine-tuning then predictions pipeline back-to-back."""
    if not app_config.fine_tune_enabled:
        raise HTTPException(status_code=400, detail="Fine-tuning disabled")

    def _run_full():
        from ..ml.local_pipeline import LocalPipeline
        pipeline = LocalPipeline()
        try:
            logger.info("Full pipeline: starting fine-tuning...")
            pipeline.run_fine_tuning()
            logger.info("Full pipeline: fine-tuning done, starting Markov pipeline...")
            pipeline.run_full_pipeline()
            logger.info("Full pipeline: complete")
        except Exception as e:
            logger.error(f"Full pipeline failed: {e}")

    background_tasks.add_task(_run_full)
    return {"status": "Full pipeline started (fine-tune → predictions)"}


@app.post("/api/fine-tune/upload-predictions")
async def upload_pythia_predictions(request: Request):
    """Receive Pythia predictions JSON from Mac after fine-tuning.

    Requires X-API-Key header matching KALSHI_API_KEY for basic auth.

    Usage from Mac:
        curl -X POST http://<pi-ip>:8000/api/fine-tune/upload-predictions \
             -H "Content-Type: application/json" \
             -H "X-API-Key: <your-kalshi-api-key>" \
             -d @data/predictions/predictions_pythia.json
    """
    # Basic auth: check shared secret
    api_key = request.headers.get('X-API-Key', '')
    expected = app_config.kalshi_api_key
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        data = await request.json()
        if 'term_predictions' not in data:
            raise HTTPException(status_code=400, detail="Missing 'term_predictions' field")

        pred_path = os.path.join('data', 'predictions', 'predictions_pythia.json')
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, 'w') as f:
            json.dump(data, f, indent=2)

        n_preds = len(data.get('term_predictions', []))
        logger.info(f"Received {n_preds} Pythia predictions from Mac")

        return {
            "status": "ok",
            "predictions_saved": n_preds,
            "message": "Predictions will be blended on next pipeline run",
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")


@app.get("/api/fine-tune/download-db")
def download_db(request: Request):
    """Download the SQLite database for fine-tuning on Mac.

    Requires X-API-Key header matching KALSHI_API_KEY for basic auth.

    Usage from Mac:
        curl -H "X-API-Key: <key>" http://<pi-ip>:8000/api/fine-tune/download-db -o data/trading_bot.db
    """
    api_key = request.headers.get('X-API-Key', '')
    expected = app_config.kalshi_api_key
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    db_path = os.path.join('data', 'trading_bot.db')
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")

    return FileResponse(
        db_path,
        media_type='application/octet-stream',
        filename='trading_bot.db',
    )


# --- TrumpGPT prompt endpoint ---

class PromptRequest(BaseModel):
    prompt: str
    word_count: int = 500
    scenario: Optional[str] = None
    temperature: float = 1.0
    qa_mode: bool = False


@app.post("/api/trumpgpt/generate")
def generate_trumpgpt(req: PromptRequest):
    """Generate text from TrumpGPT Markov chain."""
    temp = max(0.3, min(2.0, req.temperature))

    from ..ml.markov_trainer import MarkovChainTrainer
    trainer = MarkovChainTrainer(order=app_config.markov_order)
    if req.prompt.strip():
        text = trainer.generate_from_prompt(req.prompt, word_count=req.word_count, temperature=temp, qa_mode=req.qa_mode)
    else:
        text = trainer.generate_speech(scenario_type=req.scenario or 'rally', word_count=req.word_count, temperature=temp)

    if not text:
        raise HTTPException(status_code=500, detail="No trained model found. Run the pipeline first.")

    return {
        'text': text,
        'word_count': len(text.split()),
        'prompt': req.prompt,
        'scenario': req.scenario,
    }


# --- Model accuracy endpoint (1C) ---

@app.get("/api/model/accuracy")
def get_model_accuracy():
    """1C: Get prediction accuracy metrics against settled markets."""
    return predictor.evaluate_accuracy()


@app.get("/api/system/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_mode": app_config.pipeline_mode,
        "live_monitoring": live_monitor.is_monitoring,
        "config": app_config.get_status(),
    }


@app.get("/api/system/health-detailed")
def detailed_health_check():
    """Comprehensive health check for autonomous monitoring."""
    import psutil

    checks = {}

    # Database
    try:
        with get_session() as session:
            from ..database.models import Speech, Market, Term
            checks['database'] = {
                'status': 'ok',
                'speeches': session.query(Speech).count(),
                'markets': session.query(Market).count(),
                'terms': session.query(Term).count(),
            }
    except Exception as e:
        checks['database'] = {'status': 'error', 'error': str(e)}

    # Kalshi API
    checks['kalshi'] = {
        'configured': bool(app_config.kalshi_api_key),
    }

    # Fine-tuning
    ft_check = {
        'enabled': app_config.fine_tune_enabled,
        'model': app_config.fine_tune_model,
        'pytorch_available': False,
    }
    try:
        import torch
        ft_check['pytorch_available'] = True
        ft_check['torch_version'] = torch.__version__
    except ImportError:
        pass
    checks['fine_tuning'] = ft_check

    # Social media
    try:
        from ..scraper.social_media_importer import SocialMediaImporter
        importer = SocialMediaImporter()
        checks['social_media'] = importer.get_stats()
    except Exception as e:
        checks['social_media'] = {'error': str(e)}

    # Social trends
    try:
        from ..ml.social_media_analyzer import social_media_analyzer
        trends = social_media_analyzer.get_all_trends()
        checks['social_trends'] = {
            'terms_tracked': trends['total_terms'],
            'last_refresh': trends['last_refresh'],
        }
    except Exception as e:
        checks['social_trends'] = {'error': str(e)}

    # Email
    checks['email'] = {
        'configured': app_config.validate_email(),
    }

    # Disk space
    disk = psutil.disk_usage('/')
    checks['disk'] = {
        'used_gb': round(disk.used / (1024**3), 2),
        'total_gb': round(disk.total / (1024**3), 2),
        'percent': disk.percent,
        'warning': disk.percent > 85,
    }

    # RAM
    ram = psutil.virtual_memory()
    checks['memory'] = {
        'used_gb': round(ram.used / (1024**3), 2),
        'total_gb': round(ram.total / (1024**3), 2),
        'percent': ram.percent,
        'warning': ram.percent > 85,
    }

    # Predictions freshness
    pred_path = os.path.join('data', 'predictions', 'predictions_latest.json')
    if os.path.exists(pred_path):
        age_hours = (time.time() - os.path.getmtime(pred_path)) / 3600
        checks['predictions'] = {
            'status': 'ok' if age_hours < 12 else 'stale',
            'age_hours': round(age_hours, 1),
        }
    else:
        checks['predictions'] = {'status': 'missing'}

    overall = all(
        c.get('status') != 'error'
        for c in checks.values()
        if isinstance(c, dict) and 'status' in c
    )

    return {
        'status': 'ok' if overall else 'degraded',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': checks,
    }


@app.get("/api/social-media/trends")
def get_social_trends():
    """Get social media trending term scores."""
    try:
        from ..ml.social_media_analyzer import social_media_analyzer
        trends = social_media_analyzer.get_all_trends()
        # Sort by score descending
        sorted_scores = sorted(
            trends['scores'].items(),
            key=lambda x: x[1], reverse=True
        )
        return {
            'trends': [
                {'term': t, 'score': s, 'trending': s > 0.6}
                for t, s in sorted_scores[:50]
            ],
            'last_refresh': trends['last_refresh'],
            'total_terms': trends['total_terms'],
        }
    except Exception as e:
        return {'error': str(e), 'trends': []}


# --- Static frontend ---

@app.get("/")
def serve_dashboard():
    """Serve the HTML dashboard at root."""
    return FileResponse(os.path.join('static', 'index.html'))


app.mount("/static", StaticFiles(directory="static"), name="static")
