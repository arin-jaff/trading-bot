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
from .config import config


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

    # Pipeline: local or Colab depending on config
    if config.pipeline_mode == 'local':
        from .ml.local_pipeline import LocalPipeline
        local_pipeline = LocalPipeline()

        # Local pipeline runs every N hours (default 6)
        retrain_hours = config.retrain_interval_hours
        scheduler.add_job(
            _run_local_pipeline, IntervalTrigger(hours=retrain_hours),
            args=[local_pipeline, speech_scraper, term_analyzer],
            id='local_pipeline', replace_existing=True,
            name=f'Local training pipeline (every {retrain_hours}h)',
        )
        logger.info(f"Pipeline mode: LOCAL (retrain every {retrain_hours}h)")
    else:
        from .ml.colab_pipeline import ColabPipeline
        colab_pipeline = ColabPipeline()

        # Colab pipeline: export + upload + trigger training daily at 4 AM UTC
        scheduler.add_job(
            _run_colab_pipeline, CronTrigger(hour=4, minute=0),
            args=[colab_pipeline, speech_scraper, term_analyzer],
            id='colab_pipeline', replace_existing=True,
            name='Auto-retrain pipeline (daily)',
        )

        # Poll for Colab training completion every 15 minutes
        scheduler.add_job(
            _poll_colab_results, IntervalTrigger(minutes=15),
            args=[colab_pipeline],
            id='colab_poll', replace_existing=True,
            name='Poll Colab for training results',
        )
        logger.info("Pipeline mode: COLAB (daily @ 4AM UTC)")

    # Daily email digest at 8 AM UTC
    scheduler.add_job(
        _send_daily_digest, CronTrigger(hour=8, minute=0),
        id='daily_digest', replace_existing=True,
        name='Send daily email digest',
    )

    # Arbitrage scan every 10 minutes
    scheduler.add_job(
        _scan_arbitrage, IntervalTrigger(minutes=10),
        args=[trading_bot],
        id='arbitrage_scan', replace_existing=True,
        name='Scan for arbitrage opportunities',
    )

    return scheduler


def _sync_markets(client: KalshiClient, sync: MarketSync):
    try:
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


def _run_local_pipeline(pipeline, scraper: SpeechScraper,
                        analyzer: TermAnalyzer):
    """Periodic job: scrape latest speeches, then train locally."""
    try:
        scraper.scrape_all_sources()
        analyzer.process_all_unprocessed()
        result = pipeline.run_full_pipeline()
        logger.info(f"Local pipeline result: {result}")
    except Exception as e:
        logger.error(f"Scheduled local pipeline failed: {e}")


def _run_colab_pipeline(pipeline, scraper: SpeechScraper,
                        analyzer: TermAnalyzer):
    """Daily job (Colab mode): scrape then export → upload → trigger → poll."""
    try:
        scraper.scrape_all_sources()
        analyzer.process_all_unprocessed()
        result = pipeline.run_full_pipeline()
        logger.info(f"Colab pipeline result: {result}")
    except Exception as e:
        logger.error(f"Scheduled Colab pipeline failed: {e}")


def _poll_colab_results(pipeline):
    """Periodic job: check if Colab training completed, import results."""
    try:
        result = pipeline.check_and_import()
        if result.get('status') == 'imported':
            logger.info(f"Imported Colab results: {result}")
    except Exception as e:
        logger.error(f"Colab poll failed: {e}")


def _send_daily_digest():
    """Send daily email digest with portfolio and activity summary."""
    try:
        from .notifications.email_notifier import email_notifier
        if email_notifier.enabled:
            email_notifier.send_daily_digest()
    except Exception as e:
        logger.error(f"Daily digest email failed: {e}")


def _scan_arbitrage(bot: TradingBot):
    """Scan for arbitrage opportunities and auto-execute if enabled."""
    try:
        opportunities = bot.find_arbitrage()
        if opportunities:
            from .alerts import alert_manager
            for opp in opportunities:
                alert_manager.add_alert(
                    'trade_signal',
                    f"Arbitrage: {opp['market_ticker']}",
                    f"{opp['type']}: {opp['action']} (profit: ${opp['guaranteed_profit']:.4f})",
                    severity='warning',
                    data=opp,
                )
    except Exception as e:
        logger.error(f"Arbitrage scan failed: {e}")
