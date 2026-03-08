"""FastAPI backend server for the trading bot."""

import os
from datetime import datetime
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
                'terms': [t.term for t in m.terms],
            }
            for m in markets
        ]


@app.post("/api/markets/sync")
def sync_markets(background_tasks: BackgroundTasks):
    """Trigger market sync from Kalshi."""
    background_tasks.add_task(_run_market_sync)
    return {"status": "sync started"}


def _run_market_sync():
    try:
        kalshi_client.login()
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


@app.get("/api/system/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
