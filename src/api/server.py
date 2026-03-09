"""FastAPI backend server for the trading bot."""

import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from ..database.db import init_db, get_session
from ..database.models import Term, Market, TrumpEvent, Trade, TermPrediction
from ..kalshi.client import KalshiClient
from ..kalshi.market_sync import MarketSync
from ..kalshi.trading_bot import TradingBot
from ..scraper.speech_scraper import SpeechScraper
from ..scraper.term_analyzer import TermAnalyzer
from ..scraper.event_tracker import EventTracker
from ..ml.predictor import TermPredictor
from ..ml.model_trainer import ModelTrainer
from ..ml.colab_integration import ColabPredictor
from ..ml.colab_pipeline import ColabPipeline
from ..ml.drive_sync import DriveSync
from ..scraper.live_monitor import LiveSpeechMonitor
from ..alerts import alert_manager
from ..config import config as app_config

app = FastAPI(title="Trump Mentions Trading Bot", version="1.0.0")

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
model_trainer = ModelTrainer()
colab_predictor = ColabPredictor()
colab_pipeline = ColabPipeline()
drive_sync = DriveSync()
live_monitor = LiveSpeechMonitor()


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Database initialized")


# --- Market endpoints ---

@app.get("/api/markets")
def get_markets(status: Optional[str] = None):
    """Get all tracked markets."""
    with get_session() as session:
        query = session.query(Market)
        if status:
            query = query.filter(Market.status == status)
        markets = query.order_by(Market.close_time).all()
        return [
            {
                'id': m.id,
                'ticker': m.kalshi_ticker,
                'title': m.title,
                'subtitle': m.subtitle,
                'status': m.status,
                'yes_price': m.yes_price,
                'no_price': m.no_price,
                'volume': m.volume,
                'close_time': m.close_time.isoformat() if m.close_time else None,
                'expiration_time': m.expiration_time.isoformat() if m.expiration_time else None,
                'result': m.result,
                'terms': [t.term for t in m.terms],
            }
            for m in markets
        ]


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
        # Market data is public, no auth needed
        stats = market_sync.sync_markets()
        logger.info(f"Market sync: {stats}")
    except Exception as e:
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
        stats = speech_scraper.scrape_all_sources()
        logger.info(f"Speech scrape: {stats}")
        # Auto-analyze after scraping
        processed = term_analyzer.process_all_unprocessed()
        logger.info(f"Processed {processed} speeches")
    except Exception as e:
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
        preds = predictor.predict_all_terms()
        predictor.save_predictions(preds)
        logger.info(f"Generated {len(preds)} predictions")
    except Exception as e:
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
        # Try auth for trading features, but market data works without it
        kalshi_client.login()
        market_sync.sync_markets()
        speech_scraper.scrape_all_sources()
        term_analyzer.process_all_unprocessed()
        event_tracker.update_events()
        preds = predictor.predict_all_terms()
        predictor.save_predictions(preds)
        logger.info("Full refresh complete")
    except Exception as e:
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
def train_models(background_tasks: BackgroundTasks):
    """Trigger model training."""
    background_tasks.add_task(_train_models)
    return {"status": "training started"}


def _train_models():
    try:
        results = model_trainer.train()
        logger.info(f"Model training results: {results}")
    except Exception as e:
        logger.error(f"Model training failed: {e}")


@app.get("/api/ml/info")
def get_model_info():
    """Get information about trained models."""
    return model_trainer.get_model_info()


@app.get("/api/ml/predictions")
def get_ml_predictions():
    """Get ML model predictions."""
    return model_trainer.predict()


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
    return {
        'pipeline': colab_pipeline.get_status(),
        'drive': drive_sync.get_status(),
    }


@app.post("/api/pipeline/run")
def run_pipeline(background_tasks: BackgroundTasks, force: bool = False):
    """Trigger the full automated pipeline: export → upload → trigger Colab → poll → import."""
    colab_pipeline.run_pipeline_async(force=force)
    return {"status": "pipeline started", "force": force}


@app.post("/api/pipeline/export-upload")
def pipeline_export_upload(background_tasks: BackgroundTasks):
    """Export training data and upload to Google Drive (no training trigger)."""
    background_tasks.add_task(_run_export_upload)
    return {"status": "export and upload started"}


def _run_export_upload():
    try:
        result = colab_pipeline.export_and_upload()
        logger.info(f"Export & upload result: {result}")
    except Exception as e:
        logger.error(f"Export & upload failed: {e}")


@app.post("/api/pipeline/trigger-training")
def trigger_training():
    """Upload latest data to Drive and write a trigger file for Colab."""
    upload = drive_sync.upload_training_data()
    if 'error' in upload:
        raise HTTPException(status_code=500, detail=upload['error'])

    trigger = drive_sync.write_trigger_file(
        trigger_type='manual',
        extra_data={'triggered_from': 'api'},
    )
    return {'upload': upload, 'trigger': trigger}


@app.post("/api/pipeline/poll")
def poll_colab_results():
    """Check if Colab training has completed and import results."""
    return colab_pipeline.check_and_import()


@app.get("/api/drive/status")
def get_drive_status():
    """Get Google Drive integration status."""
    return drive_sync.get_status()


@app.post("/api/drive/upload")
def upload_to_drive():
    """Upload training exports to Google Drive."""
    return drive_sync.upload_training_data()


@app.post("/api/drive/download-predictions")
def download_predictions_from_drive():
    """Download latest predictions from Google Drive."""
    return drive_sync.download_predictions()


# --- System endpoints ---

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

@app.get("/api/system/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "live_monitoring": live_monitor.is_monitoring,
        "config": app_config.get_status(),
    }
