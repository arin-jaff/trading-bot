# CLAUDE.md — Trump Mentions Trading Bot

## What This Project Is

A fully automated trading system that predicts which words/phrases Donald Trump will say in upcoming speeches, trades on those predictions via Kalshi prediction markets, and continuously improves its model by scraping new speech data.

Runs autonomously on a Raspberry Pi 4. The core loop: **scrape speeches → train Markov chain → Monte Carlo simulate → predict term probabilities → trade on Kalshi**.

## Architecture Overview

### Local Mode (default — Raspberry Pi)

```
┌─────────────────── RASPBERRY PI 4 ──────────────────────────────┐
│                                                                   │
│  Scheduler (APScheduler)                                         │
│    • Market sync ────────── every 5 min (Kalshi API)             │
│    • Speech scraping ─────── every 2 hours (10 sources)          │
│    • Event tracking ──────── every 30 min                        │
│    • Predictions ─────────── every 15 min                        │
│    • Trading checks ──────── every 5 min                         │
│    • Local pipeline ──────── every 6 hours (train + simulate)    │
│    • Arbitrage scan ──────── every 10 min                        │
│    • Daily email digest ──── 8 AM UTC                            │
│    • Live speech monitor ─── every 1 min                         │
│                                                                   │
│  API: FastAPI on :8000 ──── GUI: Streamlit on :8501              │
│  Database: SQLite (SQLAlchemy ORM)                               │
│  Email: SMTP notifications                                       │
└──────────────────────────────────────────────────────┬───────────┘
                                                       │ Kalshi API
                                                       ▼
                                            ┌──── Kalshi ────────┐
                                            │  RSA key-pair auth │
                                            │  Market data (pub) │
                                            │  Order execution   │
                                            └────────────────────┘
```

### Colab Mode (optional — set PIPELINE_MODE=colab)

When more GPU power is needed, the bot can use Google Colab instead of local training.
Colab/Drive code is kept dormant in local mode but ready to reactivate.

```
Pi ──► Google Drive ──► Colab (A100 GPU) ──► Drive ──► Pi
       export data       LoRA fine-tune       predictions
                         Llama-3.1-8B
                         1000 Monte Carlo
```

## Directory Structure

```
src/
  config.py              # Central config from env vars (singleton)
  scheduler.py           # APScheduler job definitions (10+ jobs)
  alerts.py              # AlertManager — desktop + email notifications

  api/
    server.py            # FastAPI backend — 40+ endpoints

  gui/
    dashboard.py         # Streamlit dashboard — 11 tabs

  database/
    db.py                # SQLAlchemy engine, get_session() context manager
    models.py            # 10 ORM models + market_terms association table

  kalshi/
    client.py            # KalshiClient — RSA auth, rate limiting, REST API
    market_sync.py       # MarketSync — syncs markets, extracts terms from titles
    trading_bot.py       # TradingBot — Kelly criterion, risk limits, arbitrage

  scraper/
    speech_scraper.py    # SpeechScraper — 10 sources (see below)
    term_analyzer.py     # TermAnalyzer — extracts term occurrences from transcripts
    event_tracker.py     # EventTracker — discovers upcoming Trump events
    live_monitor.py      # LiveSpeechMonitor — real-time term detection (YT/CSPAN/WH)

  ml/
    markov_trainer.py       # MarkovChainTrainer — local Markov chain + Monte Carlo
    local_pipeline.py       # LocalPipeline — train → simulate → predict (Pi mode)
    predictor.py            # TermPredictor — weighted ensemble (5 signals)
    colab_integration.py    # ColabPredictor — loads predictions + Poisson correction
    feature_engineering.py  # FeatureEngineer — builds per-term feature DataFrames
    model_trainer.py        # ModelTrainer — GBM, RF, LogReg with calibration
    data_exporter.py        # DataExporter — exports corpus/context/events
    drive_sync.py           # DriveSync — Google Drive (dormant in local mode)
    colab_pipeline.py       # ColabPipeline — Colab automation (dormant in local mode)

  notifications/
    email_notifier.py    # EmailNotifier — trade alerts, daily digest, critical alerts

deploy/
  setup-pi.sh            # One-command Raspberry Pi setup
  trumpbot-api.service   # systemd service for API + scheduler
  trumpbot-gui.service   # systemd service for Streamlit dashboard
  watchdog.sh            # Health check cron

secrets/                           # git-ignored — all private keys & credentials
  kalshi_private_key.pem           # Kalshi RSA key (generate at kalshi.com)
  trump-mentions-*.json            # GCP service account key for Drive API

notebooks/
  01_finetune_trump_llm.ipynb    # LoRA fine-tuning (Colab mode only)
  02_monte_carlo_predictor.ipynb # Monte Carlo simulation (Colab mode only)

data/
  exports/          # Exported training data (.jsonl, .json)
  models/           # Markov chain pickle files (markov_v1.0.0.pkl, etc.)
  predictions/      # predictions_latest.json + timestamped copies

tests/
  test_predictor.py        # Kelly criterion and prediction tests
  test_term_extraction.py  # Term extraction from Kalshi market titles
```

## Database Models (SQLAlchemy, SQLite)

- **Market** — Kalshi market: ticker, prices, volume, status, close_time
- **Term** — tracked word/phrase: normalized form, compound flag, trend score
- **Speech** — scraped transcript: source, title, type, date, word_count, is_processed
- **TermOccurrence** — join of Term+Speech: count, context_snippets (JSON)
- **TrumpEvent** — upcoming appearance: type, location, time, is_live, topics
- **TermPrediction** — ML prediction: probability, confidence, model_name, features_used
- **PriceSnapshot** — historical market prices
- **Trade** — executed trades: side, quantity, price, pnl, strategy
- **ModelVersion** — model iteration: version string, corpus size, training duration, metrics, is_active
- **BotConfig** — persistent key/value config store
- **market_terms** — many-to-many Market↔Term association

## Speech Scraper Sources (10 total)

Defined in `src/scraper/speech_scraper.py`, method `scrape_all_sources()`:

| # | Source Key | Method | What It Gets |
|---|-----------|--------|-------------|
| 1 | `rev_transcripts` | `scrape_rev_transcripts` | Human-verified transcripts from Rev.com `/blog/transcript-category/donald-trump-transcripts` (path pagination `/page/N/`, 2s rate limit) |
| 2 | `google_news_rss` | `scrape_google_news_rss` | Meta-source: Google News RSS for transcript links across outlets |
| 3 | `whitehouse` | `scrape_whitehouse_remarks` | Official remarks from `/briefing-room/speeches-remarks/` (path pagination `/page/N/`) |
| 4 | `rollcall_factbase` | `scrape_rollcall_factbase` | rollcall.com/factbase (replaced dead factba.se) |
| 5 | `cspan` | `scrape_cspan` | C-SPAN video search (metadata, not transcripts) |
| 6 | `cspan_transcripts` | `scrape_cspan_transcripts` | C-SPAN video library with transcript extraction (personid=92774) |
| 7 | `youtube_transcripts` | `scrape_youtube_channels` | YouTube Data API + youtube-transcript-api (needs YOUTUBE_API_KEY) |
| 8 | `youtube_yt_dlp` | `scrape_youtube_yt_dlp` | yt-dlp auto-subtitle extraction (no API key needed) |
| 9 | `presidency_project` | `scrape_presidency_project` | UCSB American Presidency Project (person2=200301, `.field-docs-content`) |
| 10 | `twitter_archive` | `scrape_trump_twitter_archive` | thetrumparchive.com historical tweets (JSON/embedded data) |

## ML Prediction Pipeline

### Ensemble Predictor (TermPredictor)

Weighted ensemble of 5 signals in `src/ml/predictor.py`:
- `frequency` (0.20) — recency-weighted historical speech frequency (60-day half-life)
- `temporal` (0.10) — day-of-week and seasonal patterns
- `trend` (0.15) — recent usage velocity and acceleration
- `event_correlation` (0.15) — how event type correlates with term usage
- `monte_carlo` (0.40) — Markov chain or Colab LLM Monte Carlo predictions ← **dominant signal**

### Local Training Pipeline (default — PIPELINE_MODE=local)

`src/ml/local_pipeline.py` + `src/ml/markov_trainer.py`:

1. Checks `should_retrain()` (≥5 new speeches since last training)
2. Trains order-3 word-level Markov chain on all speech transcripts (~5 seconds)
3. Runs 2,000 Monte Carlo simulations across 5 scenario types (rally=5000w, press_conference=2000w, chopper_talk=800w, fox_interview=1500w, social_media=300w)
4. Counts term occurrences, applies Poisson length correction
5. Saves `predictions_latest.json` and imports to DB
6. Creates `ModelVersion` record (TrumpGPT v1.0.X)

Scheduled every 6 hours (configurable via `RETRAIN_INTERVAL_HOURS`).

### Colab Training Pipeline (optional — PIPELINE_MODE=colab)

`src/ml/colab_pipeline.py` orchestrates: **export → upload → trigger → poll → import**

1. **Phase 1** (`01_finetune_trump_llm.ipynb`): LoRA fine-tune Llama-3.1-8B on speech corpus using Unsloth + TRL SFTTrainer. 4-bit quantized, rank-64 LoRA, 3 epochs, 2048 max seq length.
2. **Phase 2** (`02_monte_carlo_predictor.ipynb`): Load fine-tuned model, run 1,000 simulated speeches, compute term probabilities. Output: `predictions_latest.json`.

Drive/Colab code is kept dormant in local mode — switch by setting `PIPELINE_MODE=colab`.

### Model Versioning

Each training run creates a `ModelVersion` record with:
- Version string (auto-increment patch: 1.0.0, 1.0.1, ...)
- Corpus size (speeches + word count)
- Training duration
- Simulation count and prediction count
- Artifact path (pickle file)

View all versions via `GET /api/model/versions` or the **Model Versions** dashboard tab.

## Trading Bot

`src/kalshi/trading_bot.py` — `TradingBot` class:
- **Kelly criterion** position sizing (half-Kelly for safety, capped at 25% of bankroll)
- Risk limits: `max_position_size` (100), `max_daily_loss` ($50), `max_total_exposure` ($500)
- **Cooldown**: 2-hour trading pause at 50% of daily loss limit
- **Drawdown protection**: halts trading if balance drops 30% from peak
- Edge threshold: only trades when predicted probability differs 5%+ from market price
- **Arbitrage scanner**: detects YES+NO mispricing (spread < $0.98 or > $1.02)
- `auto_trade` flag for fully automated execution
- `paper_mode` flag for simulated trading (default: on)
- Authentication: RSA key-pair signing (KALSHI_API_KEY + KALSHI_PRIVATE_KEY_PATH)

## Email Notifications

`src/notifications/email_notifier.py`:
- **Trade alerts** — sent on each trade execution
- **Daily digest** — 8 AM UTC, includes P&L, trades, top predictions
- **Critical alerts** — loss limits, training failures, drawdown events

## Environment Variables

```bash
# Kalshi API (required for trading)
KALSHI_API_KEY=
KALSHI_PRIVATE_KEY_PATH=secrets/kalshi_private_key.pem
KALSHI_BASE_URL=https://trading-api.kalshi.com/trade-api/v2

# Email Notifications
EMAIL_ENABLED=true
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_FROM=you@example.com
EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
EMAIL_TO=you@example.com

# Pipeline Mode
PIPELINE_MODE=local                    # 'local' (Pi) or 'colab' (GPU cloud)
MARKOV_ORDER=3
MONTE_CARLO_SIMULATIONS=2000
RETRAIN_INTERVAL_HOURS=6

# Google Drive (only needed if PIPELINE_MODE=colab)
GOOGLE_DRIVE_FOLDER_ID=
GOOGLE_SERVICE_ACCOUNT_KEY_PATH=secrets/<your-key>.json

# Optional
YOUTUBE_API_KEY=               # For YouTube Data API scraper
GEMINI_API_KEY=                # For current events enrichment
DATABASE_URL=sqlite:///data/trading_bot.db
LOG_LEVEL=INFO
REFRESH_INTERVAL_SECONDS=60
API_PORT=8000
GUI_PORT=8501
```

## Running the Project

```bash
make install      # pip install -r requirements.txt + spacy model
make install-pi   # Lightweight Pi dependencies (no torch/transformers)
make init         # Initialize database
make api          # Start FastAPI API server on :8000
make gui          # Start Streamlit dashboard on :8501
make all          # Start both API + GUI
make deploy-pi    # Run Raspberry Pi setup script
make export-colab # Export training data for Colab
make clean        # Delete DB + __pycache__
```

Entry points:
- `run_api.py` — initializes DB, starts scheduler, runs FastAPI on 0.0.0.0:8000
- `run_gui.py` — launches Streamlit on :8501 (connects to FastAPI backend)
- `export_for_colab.py` — one-shot export of training data to `data/exports/`

## API Endpoints (FastAPI)

All endpoints prefixed with `/api/`:

**Markets:** `GET /markets`, `GET /markets/weekly-payouts`, `POST /markets/sync`
**Terms:** `GET /terms`, `GET /terms/{id}/history`, `GET /terms/report`
**Speeches:** `POST /speeches/scrape`, `GET /speeches/stats`
**Events:** `GET /events`, `GET /events/live`, `POST /events/update`
**Predictions:** `GET /predictions`, `POST /predictions/generate`, `GET /predictions/final`
**Trading:** `GET /trading/suggestions`, `GET /trading/portfolio`, `GET|PUT /trading/config`, `POST /trading/execute`
**Trades:** `GET /trades/history` (paginated, with P&L summary)
**Kalshi Auth:** `POST /kalshi/login`
**ML:** `POST /ml/train`, `GET /ml/info`, `GET /ml/predictions`
**Model:** `GET /model/status`, `GET /model/versions`
**Colab:** `GET /colab/predictions`, `POST /colab/import`, `POST /colab/save-to-db`, `GET /colab/discovered-phrases`
**Pipeline:** `GET /pipeline/status`, `GET /pipeline/training-status`, `POST /pipeline/run`, `POST /pipeline/export-upload`, `POST /pipeline/trigger-training`, `POST /pipeline/poll`
**Drive:** `GET /drive/status`, `POST /drive/upload`, `POST /drive/download-predictions` (disabled in local mode)
**Live Monitor:** `POST /live/start`, `POST /live/stop`, `GET /live/status`
**System:** `POST /system/full-refresh`, `GET /system/health`, `GET /system/hardware`
**Alerts:** `GET /alerts`, `GET /alerts/count`, `POST /alerts/{id}/read`
**Config:** `GET /config/status`

## Key Dependencies

- **Web framework:** FastAPI + uvicorn
- **GUI:** Streamlit + Plotly
- **Database:** SQLAlchemy (SQLite)
- **Scraping:** BeautifulSoup4, requests, feedparser, trafilatura, yt-dlp, youtube-transcript-api, Playwright/Selenium
- **ML (local):** scikit-learn (GBM, RF, LogReg), pandas, numpy
- **ML (Colab):** Unsloth, TRL, transformers, bitsandbytes, PEFT
- **NLP:** spaCy, NLTK
- **Google Drive:** google-api-python-client, google-auth
- **Scheduling:** APScheduler
- **Kalshi:** requests + RSA signing (cryptography)

## Data Flow Summary

```
[10 scraper sources] → Speech table → TermAnalyzer → TermOccurrence table
                                                           ↓
                                              DataExporter → .jsonl files
                                                           ↓
                                              DriveSync → Google Drive
                                                           ↓
                                              Colab (LoRA fine-tune + Monte Carlo)
                                                           ↓
                                              predictions_latest.json → Drive
                                                           ↓
                                              ColabPipeline → TermPrediction table
                                                           ↓
                                              TermPredictor (weighted ensemble)
                                                           ↓
                                              TradingBot (Kelly criterion)
                                                           ↓
                                              Kalshi API → Trade table
```

## Known Data Quality Issues

- Many speeches in the DB have `date = 2026-03-08T20:34:*` — these are actually older speeches where `_extract_date_from_text()` failed and defaulted to `datetime.now()` during a batch scrape. The real dates should be re-extracted from titles or source pages.
- WhiteHouse.gov source was returning page chrome (100-word stubs like "Video Library") instead of actual transcripts. The scraper URL has been fixed to `/briefing-room/speeches-remarks/`.
- Roll Call/Factbase source returns news articles *about* Trump, not transcripts of his speech. These pollute the training corpus if not filtered.
- `data_exporter.py` `_chunk_transcript()` had an infinite loop bug when overlap settings caused the index to not advance — this has been fixed with a safety guard.

## Conventions

- All database access goes through `get_session()` context manager from `src/database/db.py`
- Scrapers return `int` (count of new speeches saved) and use `_save_speech()` for dedup
- Config is a singleton at `src/config.py` — access via `from .config import config`
- Logging uses `loguru` (`from loguru import logger`) everywhere
- Tests use `unittest` — run with `python -m pytest tests/`
- Async background tasks in the API use FastAPI's `BackgroundTasks`
- Scheduler jobs are synchronous functions called by APScheduler's BackgroundScheduler

## Probability Compression Fix (RESOLVED)

### Original Issue

When using Colab's LLM Monte Carlo, simulations generated ~430-word snippets but real Trump speeches are 5,000-10,000+ words. This compressed all probabilities toward ~40%.

### Fixes Applied

1. **Fix A: Poisson length normalization** — Implemented in `colab_integration.py:_apply_poisson_correction()`. Reads `avg_mentions_per_speech` from predictions JSON and corrects to full speech length using `P = 1 - exp(-lambda * target_words)`. When using local Markov chain (which generates full-length speeches per scenario), the correction is skipped automatically.

2. **Fix G: Per-scenario length normalization** — The local Markov chain generates scenario-appropriate lengths (rally=5000w, press_conference=2000w, etc.), so probabilities are naturally correct for each scenario type.

3. **Dynamic snippet detection** — `_apply_poisson_correction` reads `simulation_params.avg_words_per_speech` from the predictions JSON. If simulations are already near rally length (≥80%), Poisson correction is skipped.

### Remaining Potential Improvements

- **Fix C**: Gate temporal/trend on data sufficiency (<20 occurrences → redistribute weight)
- **Fix D**: Platt scaling calibration using settled market outcomes
- **Fix F**: Market price as Bayesian prior instead of arithmetic averaging
