"""Background scheduler for periodic data updates."""

import os
from typing import Optional
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

# Global scheduler reference for pause/resume during long DB operations
_scheduler: Optional[BackgroundScheduler] = None


def pause_scheduler():
    """Pause all scheduler jobs to avoid DB lock contention during training."""
    if _scheduler and _scheduler.running:
        _scheduler.pause()
        logger.info("Scheduler paused (DB-intensive operation in progress)")


def resume_scheduler():
    """Resume scheduler jobs after DB-intensive operation completes."""
    if _scheduler and _scheduler.running:
        _scheduler.resume()
        logger.info("Scheduler resumed")


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the background scheduler."""
    global _scheduler
    scheduler = BackgroundScheduler()
    _scheduler = scheduler

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

        # Run pipeline once at startup (10s delay to let API finish booting)
        from datetime import datetime, timedelta
        scheduler.add_job(
            _run_local_pipeline,
            'date', run_date=datetime.now() + timedelta(seconds=10),
            args=[local_pipeline, speech_scraper, term_analyzer],
            id='local_pipeline_startup', replace_existing=True,
            name='Initial pipeline run at startup',
        )
        # Nightly Pi fine-tuning (Pythia-160M LoRA, ~75 min on Pi 4)
        if config.fine_tune_enabled:
            ft_hour = config.fine_tune_hour
            scheduler.add_job(
                _run_pi_fine_tuning, CronTrigger(hour=ft_hour, minute=0, timezone='America/New_York'),
                args=[local_pipeline],
                id='pi_fine_tune', replace_existing=True,
                name=f'Nightly Pi fine-tuning ({config.fine_tune_model}, {ft_hour}AM ET)',
            )
            logger.info(f"Pi fine-tuning: {config.fine_tune_model} nightly at {ft_hour}AM ET")

        logger.info(f"Pipeline mode: LOCAL (retrain every {retrain_hours}h, first run in 10s)")
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

    # Database pruning — daily cleanup of old predictions (6:30 AM ET)
    scheduler.add_job(
        _prune_old_predictions, CronTrigger(hour=6, minute=30, timezone='America/New_York'),
        id='prune_predictions', replace_existing=True,
        name='Prune old TermPredictions (keep 7 days)',
    )

    # Accuracy evaluation — run daily to recalibrate against settled markets (7 AM ET)
    scheduler.add_job(
        _evaluate_accuracy, CronTrigger(hour=7, minute=0, timezone='America/New_York'),
        args=[predictor],
        id='accuracy_eval', replace_existing=True,
        name='Evaluate prediction accuracy vs settled markets',
    )

    # Daily email digest at 8 AM ET (handles EST/EDT automatically)
    scheduler.add_job(
        _send_daily_digest, CronTrigger(hour=8, minute=0, timezone='America/New_York'),
        id='daily_digest', replace_existing=True,
        name='Send daily email digest (8 AM ET)',
    )

    # Social media scraping — every 30 min (primary terminology source)
    social_minutes = config.social_media_scrape_minutes
    scheduler.add_job(
        _scrape_social_media, IntervalTrigger(minutes=social_minutes),
        id='social_media_scrape', replace_existing=True,
        name=f'Scrape Truth Social + Twitter (every {social_minutes}m)',
    )

    # Social media term analysis — every 30 min (after scrape)
    scheduler.add_job(
        _analyze_social_trends, IntervalTrigger(minutes=social_minutes),
        id='social_analysis', replace_existing=True,
        name='Analyze social media trending terms',
    )

    # 2A: News enrichment via Gemini — every hour
    scheduler.add_job(
        _refresh_news_enrichment, IntervalTrigger(hours=1),
        id='news_enrichment', replace_existing=True,
        name='Refresh Gemini news enrichment',
    )

    # 3B: Position management (profit-taking + stop-loss) — every 5 minutes
    scheduler.add_job(
        _manage_positions, IntervalTrigger(minutes=5),
        args=[trading_bot],
        id='position_management', replace_existing=True,
        name='Manage open positions (profit-take/stop-loss)',
    )

    return scheduler


def _sync_markets(client: KalshiClient, sync: MarketSync):
    """3A: Detect new markets and alert on front-running opportunities."""
    try:
        stats = sync.sync_markets()
        new_tickers = stats.get('new_market_tickers', [])
        if new_tickers:
            logger.info(f"3A: {len(new_tickers)} new markets detected: {new_tickers}")
            from .alerts import alert_manager
            for ticker in new_tickers:
                alert_manager.add_alert(
                    'new_market',
                    f"New Market: {ticker}",
                    f"New Trump Mentions market created. "
                    f"Check for front-running opportunity — market may open at 50c.",
                    severity='warning',
                    data={'ticker': ticker},
                )
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
            from .alerts import alert_manager
            for s in suggestions[:3]:  # max 3 auto-trades per cycle
                result = bot.execute_trade(s, require_confirmation=False)
                if result and result.get('status') in ('placed', 'paper_trade'):
                    alert_manager.add_alert(
                        'trade_signal',
                        f"Trade: {result.get('side', '?').upper()} {result.get('ticker', '?')}",
                        f"{result.get('side', '?').upper()} {result.get('quantity', 0)}x "
                        f"{result.get('ticker', '?')} @ {result.get('price_cents', 0)}c",
                        severity='warning',
                        data=result,
                    )
    except Exception as e:
        logger.error(f"Scheduled trading check failed: {e}")


def _run_local_pipeline(pipeline, scraper: SpeechScraper,
                        analyzer: TermAnalyzer):
    """Periodic job: run pipeline (scraping handled by separate 2h jobs)."""
    try:
        # Process any unprocessed speeches from the last scrape cycle
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


def _prune_old_predictions():
    """Daily: delete TermPrediction rows older than 7 days."""
    try:
        from datetime import datetime, timedelta
        from .database.db import get_session
        from .database.models import TermPrediction
        cutoff = datetime.now() - timedelta(days=7)
        with get_session() as session:
            deleted = session.query(TermPrediction).filter(
                TermPrediction.created_at < cutoff,
            ).delete()
            if deleted:
                logger.info(f"Pruned {deleted} old TermPrediction rows (>7 days)")
    except Exception as e:
        logger.error(f"Prediction pruning failed: {e}")


def _evaluate_accuracy(predictor: TermPredictor):
    """Daily: evaluate prediction accuracy and log calibration metrics."""
    try:
        result = predictor.evaluate_accuracy()
        if 'brier_score' in result:
            logger.info(f"Accuracy eval: Brier={result['brier_score']:.4f}, "
                        f"hit_rate={result.get('hit_rate', 0):.1%}, "
                        f"data_points={result.get('data_points', 0)}")
    except Exception as e:
        logger.error(f"Accuracy evaluation failed: {e}")


def _scrape_social_media():
    """Periodic job: scrape latest Truth Social posts + rebuild daily digests."""
    try:
        from .scraper.social_media_importer import SocialMediaImporter
        importer = SocialMediaImporter()
        new_posts = importer.scrape_latest_posts()
        if new_posts:
            logger.info(f"Social media scrape: {new_posts} new posts")
    except Exception as e:
        logger.error(f"Social media scrape failed: {e}")


def _refresh_news_enrichment():
    """2A: Refresh current events enrichment from Gemini."""
    try:
        from .ml.news_enrichment import news_enricher
        news_enricher.refresh()
    except Exception as e:
        logger.error(f"News enrichment refresh failed: {e}")


def _manage_positions(bot: TradingBot):
    """3B: Active position management — profit-taking and stop-loss."""
    try:
        actions = bot.manage_positions()
        if actions:
            from .alerts import alert_manager
            for action in actions:
                alert_manager.add_alert(
                    'position_management',
                    f"{action['type'].replace('_', ' ').title()}: {action['ticker']}",
                    f"{action['type']}: {action['quantity']}x {action['side']} "
                    f"(entry: {action['entry_price']:.2f}, now: {action['current_price']:.2f}, "
                    f"unrealized: {action['unrealized']:+.4f})",
                    severity='info',
                    data=action,
                )
    except Exception as e:
        logger.error(f"Position management failed: {e}")


def _run_pi_fine_tuning(pipeline):
    """Nightly job: fine-tune Pythia-160M with LoRA on Pi."""
    try:
        result = pipeline.run_fine_tuning()
        if result:
            logger.info(f"Pi fine-tuning result: {result}")
    except Exception as e:
        logger.error(f"Pi fine-tuning failed: {e}")


def _analyze_social_trends():
    """Periodic job: analyze social media posts for trending terms."""
    try:
        from .ml.social_media_analyzer import social_media_analyzer
        social_media_analyzer.refresh()
    except Exception as e:
        logger.error(f"Social media analysis failed: {e}")


