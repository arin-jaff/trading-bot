"""Background scheduler for periodic data updates."""

import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from .kalshi.client import KalshiClient
from .kalshi.market_sync import MarketSync
from .kalshi.trading_bot import TradingBot
from .scraper.speech_scraper import SpeechScraper
from .scraper.term_analyzer import TermAnalyzer
from .scraper.event_tracker import EventTracker
from .ml.predictor import TermPredictor


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the background scheduler."""
    scheduler = BackgroundScheduler()

    # Initialize components
    client = KalshiClient()
    market_sync = MarketSync(client)
    speech_scraper = SpeechScraper()
    term_analyzer = TermAnalyzer()
    event_tracker = EventTracker()
    predictor = TermPredictor()
    trading_bot = TradingBot(client, predictor)

    refresh_interval = int(os.getenv('REFRESH_INTERVAL_SECONDS', '300'))

    # Market sync every 5 minutes
    scheduler.add_job(
        _sync_markets, IntervalTrigger(seconds=refresh_interval),
        args=[client, market_sync],
        id='market_sync', replace_existing=True,
        name='Sync Kalshi markets',
    )

    # Speech scraping every 2 hours
    scheduler.add_job(
        _scrape_speeches, IntervalTrigger(hours=2),
        args=[speech_scraper, term_analyzer],
        id='speech_scrape', replace_existing=True,
        name='Scrape speeches',
    )

    # Event tracking every 30 minutes
    scheduler.add_job(
        _update_events, IntervalTrigger(minutes=30),
        args=[event_tracker],
        id='event_update', replace_existing=True,
        name='Update events',
    )

    # Live status check every minute
    scheduler.add_job(
        event_tracker.check_and_update_live_status,
        IntervalTrigger(minutes=1),
        id='live_check', replace_existing=True,
        name='Check live events',
    )

    # Prediction generation every 15 minutes
    scheduler.add_job(
        _generate_predictions, IntervalTrigger(minutes=15),
        args=[predictor],
        id='predictions', replace_existing=True,
        name='Generate predictions',
    )

    # Trading suggestions every 5 minutes
    scheduler.add_job(
        _check_trading, IntervalTrigger(minutes=5),
        args=[trading_bot],
        id='trading_check', replace_existing=True,
        name='Check trading opportunities',
    )

    return scheduler


def _sync_markets(client: KalshiClient, sync: MarketSync):
    try:
        # Market data is public, no auth needed
        sync.sync_markets()
    except Exception as e:
        logger.error(f"Scheduled market sync failed: {e}")


def _scrape_speeches(scraper: SpeechScraper, analyzer: TermAnalyzer):
    try:
        scraper.scrape_all_sources()
        analyzer.process_all_unprocessed()
    except Exception as e:
        logger.error(f"Scheduled speech scrape failed: {e}")


def _update_events(tracker: EventTracker):
    try:
        tracker.update_events()
    except Exception as e:
        logger.error(f"Scheduled event update failed: {e}")


def _generate_predictions(predictor: TermPredictor):
    try:
        preds = predictor.predict_all_terms()
        predictor.save_predictions(preds)
    except Exception as e:
        logger.error(f"Scheduled prediction failed: {e}")


def _check_trading(bot: TradingBot):
    try:
        if bot.check_daily_loss_limit():
            logger.warning("Daily loss limit reached, skipping trading check")
            return

        suggestions = bot.generate_suggestions()
        if suggestions and bot.auto_trade:
            for s in suggestions[:3]:  # max 3 auto-trades per cycle
                bot.execute_trade(s, require_confirmation=False)
    except Exception as e:
        logger.error(f"Scheduled trading check failed: {e}")
