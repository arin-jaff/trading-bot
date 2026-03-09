# CLAUDE.md — Trump Mentions Trading Bot

## What This Project Is

A fully automated trading system that predicts which words/phrases Donald Trump will say in upcoming speeches, trades on those predictions via Kalshi prediction markets, and continuously improves its model by scraping new speech data and fine-tuning an LLM on Google Colab.

The core loop: **scrape speeches → extract terms → fine-tune LLM → Monte Carlo simulate → predict term probabilities → trade on Kalshi**.

## Architecture Overview

```
┌─────────────── LOCAL SERVER (FastAPI + Streamlit) ───────────────┐
│                                                                   │
│  Scheduler (APScheduler)                                         │
│    • Market sync ────────── every 5 min (Kalshi API)             │
│    • Speech scraping ─────── every 2 hours (10 sources)          │
│    • Event tracking ──────── every 30 min                        │
│    • Predictions ─────────── every 15 min                        │
│    • Trading checks ──────── every 5 min                         │
│    • Colab pipeline ──────── daily at 4 AM UTC                   │
│    • Colab poll ──────────── every 15 min                        │
│    • Live speech monitor ─── every 1 min                         │
│                                                                   │
│  API: FastAPI on :8000 ──── GUI: Streamlit on :8501              │
│  Database: SQLite (SQLAlchemy ORM)                               │
└──────────┬──────────────────────────────────────────┬────────────┘
           │ Google Drive API                         │ Kalshi API
           ▼                                          ▼
┌──── Google Drive ────┐                   ┌──── Kalshi ────────┐
│  data/               │                   │  RSA key-pair auth │
│  predictions/        │                   │  Market data (pub) │
│  triggers/           │                   │  Order execution   │
└──────────┬───────────┘                   └────────────────────┘
           │
           ▼
┌──── Google Colab Pro (A100 GPU) ────┐
│  01_finetune_trump_llm.ipynb        │
│    LoRA fine-tune Llama-3-8B        │
│  02_monte_carlo_predictor.ipynb     │
│    1000 simulated speeches → probs  │
└─────────────────────────────────────┘
```

## Directory Structure

```
src/
  config.py              # Central config from env vars (singleton)
  scheduler.py           # APScheduler job definitions
  alerts.py              # AlertManager for live events, trades, detections

  api/
    server.py            # FastAPI backend — all /api/* endpoints

  gui/
    dashboard.py         # Streamlit dashboard (connects to FastAPI)

  database/
    db.py                # SQLAlchemy engine, get_session() context manager
    models.py            # 9 ORM models + market_terms association table

  kalshi/
    client.py            # KalshiClient — RSA auth, rate limiting, REST API
    market_sync.py       # MarketSync — syncs markets, extracts terms from titles
    trading_bot.py       # TradingBot — Kelly criterion sizing, risk limits

  scraper/
    speech_scraper.py    # SpeechScraper — 10 sources (see below)
    term_analyzer.py     # TermAnalyzer — extracts term occurrences from transcripts
    event_tracker.py     # EventTracker — discovers upcoming Trump events
    live_monitor.py      # LiveSpeechMonitor — real-time term detection (YT/CSPAN/WH)

  ml/
    feature_engineering.py  # FeatureEngineer — builds per-term feature DataFrames
    model_trainer.py        # ModelTrainer — GBM, RF, LogReg with calibration
    predictor.py            # TermPredictor — weighted ensemble (5 signals)
    colab_integration.py    # ColabPredictor — loads Colab predictions (file or API)
    data_exporter.py        # DataExporter — exports corpus/context/events for Colab
    drive_sync.py           # DriveSync — Google Drive upload/download/trigger
    colab_pipeline.py       # ColabPipeline — full automation orchestrator

secrets/                           # git-ignored — all private keys & credentials
  kalshi_private_key.pem           # Kalshi RSA key (generate at kalshi.com)
  trump-mentions-*.json            # GCP service account key for Drive API
  README.md                        # Setup instructions for credentials

notebooks/
  01_finetune_trump_llm.ipynb    # LoRA fine-tuning (Unsloth + TRL)
  02_monte_carlo_predictor.ipynb # Monte Carlo simulation → term probabilities

data/
  exports/          # Exported training data (.jsonl, .json)
  models/           # Local model artifacts and metrics
  predictions/      # Prediction JSON files from Colab

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

### Local Prediction (TermPredictor)

Weighted ensemble of 5 signals in `src/ml/predictor.py`:
- `frequency` (0.20) — historical speech frequency of the term
- `temporal` (0.10) — recency-weighted usage trends
- `market_sentiment` (0.15) — current Kalshi market prices
- `event_correlation` (0.15) — how event type correlates with term usage
- `colab_monte_carlo` (0.40) — Colab fine-tuned LLM predictions ← **dominant signal**

### Colab Training Pipeline

1. **Phase 1** (`01_finetune_trump_llm.ipynb`): LoRA fine-tune Llama-3.1-8B on speech corpus using Unsloth + TRL SFTTrainer. 4-bit quantized, rank-64 LoRA, 3 epochs, 2048 max seq length.
2. **Phase 2** (`02_monte_carlo_predictor.ipynb`): Load fine-tuned model, run 1,000 simulated speeches for an event context, extract n-grams, compute term probabilities. Output: `predictions_latest.json`.

### Automated Pipeline (ColabPipeline)

`src/ml/colab_pipeline.py` orchestrates: **export → upload → trigger → poll → import**

- Checks `should_retrain()` (≥5 new speeches since last training)
- Exports via `DataExporter` (corpus .jsonl + term context + event pairs)
- Uploads to Google Drive via `DriveSync`
- Writes `training_trigger.json` to Drive's `triggers/` subfolder
- Polls for `training_complete.json` (2-hour timeout, 2-min interval)
- Downloads `predictions_latest.json` and saves to DB

Scheduled daily at 4 AM UTC + poll every 15 min for results.

## Trading Bot

`src/kalshi/trading_bot.py` — `TradingBot` class:
- **Kelly criterion** position sizing (capped at 25% of bankroll per bet)
- Risk limits: `max_position_size`, `max_daily_loss`, `max_total_exposure`
- Edge threshold: only trades when predicted probability differs enough from market price
- `auto_trade` flag for fully automated execution
- Authentication: RSA key-pair signing (KALSHI_API_KEY + KALSHI_PRIVATE_KEY_PATH)

## Environment Variables

```bash
# Kalshi API (required for trading)
KALSHI_API_KEY=
KALSHI_PRIVATE_KEY_PATH=secrets/kalshi_private_key.pem
KALSHI_BASE_URL=https://trading-api.kalshi.com/trade-api/v2

# Colab integration
COLAB_PREDICTION_URL=          # ngrok URL for live Colab inference API

# Google Drive automation
GOOGLE_DRIVE_FOLDER_ID=        # Shared Drive folder ID
GOOGLE_SERVICE_ACCOUNT_KEY_PATH=secrets/<your-key>.json  # Path to service account JSON key
# OR: GOOGLE_DRIVE_CREDENTIALS_JSON=  # Raw JSON (for containers)

# Optional
YOUTUBE_API_KEY=               # For YouTube Data API scraper
DATABASE_URL=sqlite:///trading_bot.db
LOG_LEVEL=INFO
REFRESH_INTERVAL_SECONDS=300
API_PORT=8000
GUI_PORT=8501
```

## Running the Project

```bash
make install    # pip install -r requirements.txt + spacy model
make init       # Initialize database
make run        # Start FastAPI API server on :8000
make gui        # Start Streamlit dashboard on :8501
make all        # Start both API + GUI
make export     # Run export_for_colab.py (exports training data)
make clean      # Delete DB + __pycache__
```

Entry points:
- `run_api.py` — initializes DB, starts scheduler, runs FastAPI on 0.0.0.0:8000
- `run_gui.py` — launches Streamlit on :8501 (connects to FastAPI backend)
- `export_for_colab.py` — one-shot export of training data to `data/exports/`

## API Endpoints (FastAPI)

All endpoints prefixed with `/api/`:

**Markets:** `GET /markets`, `POST /markets/sync`
**Terms:** `GET /terms`, `GET /terms/{id}/history`, `GET /terms/report`
**Speeches:** `POST /speeches/scrape`, `GET /speeches/stats`
**Events:** `GET /events`, `GET /events/live`, `POST /events/update`
**Predictions:** `GET /predictions`, `POST /predictions/generate`
**Trading:** `GET /trading/suggestions`, `GET /trading/portfolio`, `GET|PUT /trading/config`, `POST /trading/execute`
**Kalshi Auth:** `POST /kalshi/login`
**ML:** `POST /ml/train`, `GET /ml/info`, `GET /ml/predictions`
**Colab:** `GET /colab/predictions`, `POST /colab/import`, `POST /colab/save-to-db`, `GET /colab/discovered-phrases`
**Pipeline:** `GET /pipeline/status`, `POST /pipeline/run`, `POST /pipeline/export-upload`, `POST /pipeline/trigger-training`, `POST /pipeline/poll`
**Drive:** `GET /drive/status`, `POST /drive/upload`, `POST /drive/download-predictions`
**Live Monitor:** `POST /live/start`, `POST /live/stop`, `GET /live/status`
**System:** `POST /system/full-refresh`, `GET /system/health`
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
