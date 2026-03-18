# CLAUDE.md — Trump Mentions Trading Bot

## What This Project Is

A fully automated trading system that predicts which words/phrases Donald Trump will say in upcoming speeches, trades on those predictions via Kalshi prediction markets, and continuously improves its model by scraping new speech data.

Runs autonomously on a Raspberry Pi 4. The core loop: **scrape speeches + tweets + Truth Social → train Markov chain → Monte Carlo simulate → predict term probabilities → trade on Kalshi**.

Fine-tunes **Pythia-410M** (EleutherAI, 410M params) with LoRA on Mac, pushes predictions to Pi via API. Corpus includes ~56K historical tweets (auto-imported on first run), live Truth Social posts (scraped every 2 hours via Mastodon API), live Twitter/X posts (scraped via Nitter RSS), and 10 speech transcript sources.

> See [OPTIMIZATION.md](OPTIMIZATION.md) for the full changelog of the v2 optimization pass (fee-aware Kelly, Gemini news enrichment, correlation matrix, position management, and more).

## Architecture Overview

### Local Mode (default — Raspberry Pi)

```
┌─────────────────── RASPBERRY PI 4 ──────────────────────────────┐
│                                                                   │
│  Scheduler (APScheduler)                                         │
│    • Market sync ────────── every 5 min (Kalshi API + new mkt)   │
│    • Speech scraping ─────── every 2 hours (10 sources + dedup)  │
│    • Event tracking ──────── every 30 min                        │
│    • Predictions ─────────── every 15 min (6-signal ensemble)    │
│    • Trading checks ──────── every 5 min (fee-aware Kelly)       │
│    • Position management ─── every 5 min (profit-take/stop-loss) │
│    • News enrichment ─────── every 1 hour (Gemini Flash Lite)    │
│    • Local pipeline ──────── every 6 hours (train + simulate)    │
│    • Daily email digest ──── 8 AM UTC                            │
│    • Live speech monitor ─── every 1 min                         │
│    • Pythia blend ─────────── if predictions synced from Mac      │
│                                                                   │
│  API: FastAPI on :8000 ──── Dashboard: static HTML on :8000      │
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
  scheduler.py           # APScheduler job definitions (11+ jobs)
  alerts.py              # AlertManager — desktop + email notifications

  api/
    server.py            # FastAPI backend — 50+ endpoints

  gui/
    dashboard.py         # Streamlit dashboard — 11 tabs (legacy, optional)

  database/
    db.py                # SQLAlchemy engine, get_session() context manager
    models.py            # 10 ORM models + market_terms association table

  kalshi/
    client.py            # KalshiClient — RSA auth, rate limiting, REST API
    market_sync.py       # MarketSync — syncs markets, extracts terms from titles
    trading_bot.py       # TradingBot — Kelly criterion, risk limits, position management

  scraper/
    speech_scraper.py    # SpeechScraper — 10 sources (see below)
    social_media_importer.py  # SocialMediaImporter — bulk Twitter/Truth Social import
    term_analyzer.py     # TermAnalyzer — extracts term occurrences from transcripts
    event_tracker.py     # EventTracker — discovers upcoming Trump events
    live_monitor.py      # LiveSpeechMonitor — real-time term detection (YT/CSPAN/WH)

  ml/
    markov_trainer.py       # MarkovChainTrainer — local Markov chain + Monte Carlo
    fine_tuner.py           # GPT2FineTuner — Pi-native Pythia-410M LoRA fine-tuning
    local_pipeline.py       # LocalPipeline — 8-phase pipeline (Markov + optional fine-tune)
    predictor.py            # TermPredictor — weighted ensemble (6 signals)
    news_enrichment.py      # NewsEnricher — Gemini 2.0 Flash Lite current events
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

scripts/
  backfill_settlements.py    # One-time: match settled markets to predictions, Platt calibration
  clean_html_posts.py        # One-time: strip HTML from social media posts imported before cleaning was added

data/
  exports/          # Exported training data (.jsonl, .json)
  imports/          # Downloaded social media archives (tweets, Truth Social)
  models/           # Markov chain pickle files + GPT-2/Pythia LoRA adapters
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
- **ModelVersion** — model iteration: version string, corpus size, training duration, metrics, is_active (model_type: `markov_chain` or `gpt2_lora`)
- **BotConfig** — persistent key/value config store
- **market_terms** — many-to-many Market↔Term association

## Speech Sources

### Scraper Sources (10 total)

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

### Social Media Bulk Import

`src/scraper/social_media_importer.py` — `SocialMediaImporter` class:

- **Twitter archive (bulk, one-time)**: Downloads ~56K Trump tweets, parses JSON/CSV, saves individual posts (`speech_type='social_media'`) and groups them by date into daily digests (`speech_type='social_media_daily'`) for Markov training (minimum 50 words per digest). Auto-runs on first pipeline execution if no tweets exist in the DB.
- **Truth Social (live, every 2 hours)**: Scrapes new posts via Mastodon-compatible API (Truth Social is a Mastodon fork). New posts saved individually, daily digests rebuilt for last 7 days.
- **Twitter/X (live, every 2 hours)**: Scrapes recent tweets via Nitter RSS proxies (5 instances) with RSS Bridge fallback. No Twitter API key needed. Fills the gap after the archive (which ends ~2024-11).
- **HTML cleaning**: All posts are stripped of HTML tags, entities, URLs, and @mentions before saving. A one-time cleanup script (`scripts/clean_html_posts.py`) is provided for posts imported before cleaning was added.
- **Daily digest grouping**: Individual posts (~30 words) are too short for the Markov trainer's `word_count >= 100` filter. Posts are grouped by date into daily digests. A day with 10+ posts easily crosses 100 words. Digests are updated in-place when new posts arrive.
- Triggered via dashboard buttons (Import Twitter Archive / Scrape Truth Social) or CLI (`make import-twitter`)
- Also runs automatically: pipeline Phase 0 handles initial import + live scraping before each training cycle

## ML Prediction Pipeline

### Ensemble Predictor (TermPredictor)

Weighted ensemble of 6 signals in `src/ml/predictor.py`:
- `frequency` (0.20) — recency-weighted historical speech frequency (60-day half-life)
- `temporal` (0.05) — day-of-week and seasonal patterns; **gated out** when >30% of speech dates are poisoned (returns `None`, weight redistributed)
- `trend` (0.15) — recent usage velocity and acceleration
- `event_correlation` (0.10) — how event type correlates with term usage
- `monte_carlo` (0.40) — Markov chain or Colab LLM Monte Carlo predictions ← **dominant signal**
- `news_relevance` (0.10) — Gemini 2.0 Flash Lite current events talking points

Post-prediction processing:
- **Correlation boost (2D):** terms co-occurring with high-confidence predictions (Jaccard > 0.3) get a proportional probability boost
- **Fee-aware Kelly:** `_kelly_criterion()` subtracts 4% round-trip Kalshi fee from edge before computing bet size
- **Confidence scaling:** Kelly fraction multiplied by prediction confidence (0-1)

### Local Training Pipeline (default — PIPELINE_MODE=local)

`src/ml/local_pipeline.py` — 8-phase pipeline:

**Always runs (Phases 0-5, ~35 seconds):**

0. **Phase 0:** Social media refresh — auto-imports Twitter archive on first run (~56K tweets); scrapes latest Truth Social (Mastodon API) + Twitter/X (Nitter RSS); rebuilds recent daily digests (non-fatal if sources are down)
1. Checks `should_retrain()` (≥5 new speeches since last training)
2. **Phase 1:** Trains order-3 word-level Markov chain on all speech transcripts (~5 seconds)
3. **Phase 2:** Loads tracked terms from DB
4. **Phase 3:** Runs 2,000 Monte Carlo simulations across 5 scenario types (rally=5000w, press_conference=2000w, chopper_talk=800w, fox_interview=1500w, social_media=300w). Adjusts scenario weights based on next confirmed TrumpEvent.
5. **Phase 4-5:** Saves `predictions_latest.json`, imports to DB, creates `ModelVersion` record (TrumpGPT v1.0.X)

**Pythia blending (automatic if predictions exist on disk):**

After Phase 5, the pipeline checks if `data/predictions/predictions_pythia.json` exists (synced from Mac). If present, it blends Markov (60%) + Pythia (40%) predictions and re-imports to DB.

**Fine-tuning runs on Mac** via `scripts/fine_tune_mac.py`. After completion, predictions are pushed to the Pi via `POST /api/fine-tune/upload-predictions`. See the Mac Fine-Tuning section below.

Scheduled every 6 hours (configurable via `RETRAIN_INTERVAL_HOURS`).

### Mac Fine-Tuning

`scripts/fine_tune_mac.py` — standalone all-in-one script:

**Model: Llama-3.2-1B (Meta)**
- 1B parameters, trained on 15T tokens — dramatically better language quality than smaller models
- LoRA adapter: rank 16, ~2-3MB trainable params, targeting `q_proj`/`v_proj` (auto-detected)
- Runs on Mac (Apple Silicon or Intel) — ~1 hour on M-series, ~2-3 hours on Intel
- Configurable via `FINE_TUNE_MODEL` env var (also supports `EleutherAI/pythia-410m`, `gpt2`, etc.)

**Workflow:**
1. `scp arin@<pi-ip>:~/trading-bot/data/trading_bot.db data/trading_bot.db`
2. `python scripts/fine_tune_mac.py --pi-url http://<pi-ip>:8000`
3. Script trains, runs Monte Carlo, and POSTs predictions to Pi automatically

**Architecture auto-detection** via `_detect_lora_targets()` — supports GPT-2, Pythia/GPT-NeoX, Llama, OPT, etc. Change model with `FINE_TUNE_MODEL` env var.

**Why Llama-3.2-1B:**
- 15T token pretraining (vs Pythia's 300B) — far better language quality
- 1B params fits in ~6GB RAM on Mac with LoRA
- ~1 hour training on Apple Silicon — runs overnight via launchd
- Pi never loads the model — only reads the ~50KB predictions JSON
- PyTorch doesn't support Raspberry Pi ARM — fine-tuning must run on Mac

### Colab Training Pipeline (optional — PIPELINE_MODE=colab)

`src/ml/colab_pipeline.py` orchestrates: **export → upload → trigger → poll → import**

1. **Phase 1** (`01_finetune_trump_llm.ipynb`): LoRA fine-tune Llama-3.1-8B on speech corpus using Unsloth + TRL SFTTrainer. 4-bit quantized, rank-64 LoRA, 3 epochs, 2048 max seq length.
2. **Phase 2** (`02_monte_carlo_predictor.ipynb`): Load fine-tuned model, run 1,000 simulated speeches, compute term probabilities. Output: `predictions_latest.json`.

Drive/Colab code is kept dormant in local mode — switch by setting `PIPELINE_MODE=colab`.

### Model Versioning

Each training run creates a `ModelVersion` record with:
- Version string: Markov versions are `1.0.X` (auto-increment patch), Pythia versions are `2.0.X`
- Model type: `markov_chain` or `gpt2_lora`
- Corpus size (speeches + word count)
- Training duration
- Simulation count and prediction count
- Artifact path (pickle file for Markov, LoRA adapter directory for Pythia)
- Metrics (for Pythia: final_loss, best_loss, total_steps, trainable_params, lora_rank)

View all versions via `GET /api/model/versions` or the **Model Versions** dashboard tab.

## Trading Bot

`src/kalshi/trading_bot.py` — `TradingBot` class:
- **Fee-aware Kelly criterion** position sizing (half-Kelly, capped at 25%; subtracts 4% round-trip fee from edge)
- **Confidence scaling**: Kelly multiplied by prediction confidence (0-1)
- Risk limits: `max_position_size` (100), `max_daily_loss` ($50), `max_total_exposure` ($500), `min_volume` (50)
- **Cooldown**: 2-hour trading pause at 50% of daily loss limit
- **Drawdown protection**: halts trading if balance drops 30% from peak
- Edge threshold: only trades when `|edge| - 0.04 > 0` (net of fees)
- **Liquidity filter**: skips markets with volume < 50; caps position to 10% of market volume
- **Time-to-close decay**: reduces position size for markets closing <2h (0.7x) or >5d (0.5x)
- **Position management**: profit-taking (sell 50% when gain >8c) and stop-loss (sell all when down >15c)
- **1c/99c filter**: markets at 1c or 99c are skipped — these are "virtually certain/dead" (already said or won't be said) and untradeable
- **New market front-running**: alerts + email when new markets detected during sync
- `auto_trade` flag for fully automated execution
- `paper_mode` flag for simulated trading (default: on)
- Authentication: RSA key-pair signing (KALSHI_API_KEY + KALSHI_PRIVATE_KEY_PATH)

## Email Notifications

`src/notifications/email_notifier.py`:
- **Daily digest** — 8 AM ET, includes P&L, trades, top predictions, scraper stats

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
GEMINI_API_KEY=                # For Gemini 2.0 Flash Lite news enrichment (~$0.0024/day)
DATABASE_URL=sqlite:///data/trading_bot.db
LOG_LEVEL=INFO
REFRESH_INTERVAL_SECONDS=60
API_PORT=8000
GUI_PORT=8501
```

## Running the Project

### Quick Start (any machine)

```bash
# 1. Install dependencies
make install          # Full deps (includes spacy model)
# OR
make install-pi       # Lightweight Pi deps (no torch/transformers)

# 2. Initialize database
make init

# 3. Start the API server (includes scheduler + dashboard)
make api
# → API at http://localhost:8000
# → Dashboard at http://localhost:8000 (static HTML, served by FastAPI)

# 4. (Optional) Start Streamlit dashboard
make gui
# → Streamlit at http://localhost:8501
```

### Fine-Tuning (on Mac)

```bash
# 1. Install fine-tuning deps on your Mac
make install-finetune

# 2. Copy latest DB from Pi
scp arin@<pi-ip>:~/trading-bot/data/trading_bot.db data/trading_bot.db

# 3. Fine-tune + auto-push predictions to Pi
python scripts/fine_tune_mac.py --pi-url http://<pi-ip>:8000
```

### Raspberry Pi Deployment

```bash
# On the Pi:
git clone <repo-url>
cd trading-bot

# Copy .env and secrets from your development machine
# scp .env pi@<pi-ip>:/home/pi/trading-bot/.env
# scp -r secrets/ pi@<pi-ip>:/home/pi/trading-bot/secrets/

# Run setup script (installs deps, creates systemd services)
make deploy-pi

# Start services
sudo systemctl start trumpbot-api
sudo systemctl start trumpbot-gui   # optional — saves RAM if skipped

# Verify
curl http://localhost:8000/api/system/health

# Access dashboard from your laptop
# http://<pi-ip>:8000
```

### Make Targets

```bash
make install          # pip install -r requirements.txt + spacy model
make install-pi       # Lightweight Pi dependencies (no torch/transformers)
make install-finetune # pip install torch transformers peft datasets accelerate
make init             # Initialize database
make api              # Start FastAPI API server on :8000
make gui              # Start Streamlit dashboard on :8501
make all              # Start both API + GUI
make import-twitter   # Download + import Trump Twitter archive
make deploy-pi        # Run Raspberry Pi setup script
make export-colab     # Export training data for Colab
make clean            # Delete DB + __pycache__
```

### Entry Points

- `run_api.py` — initializes DB, starts scheduler, runs FastAPI on 0.0.0.0:8000
- `run_gui.py` — launches Streamlit on :8501 (connects to FastAPI backend)
- `export_for_colab.py` — one-shot export of training data to `data/exports/`

## Dashboard (static HTML)

The primary dashboard is a single-page app at `static/index.html`, served by FastAPI at the root URL (`http://localhost:8000`). Built with Alpine.js + Chart.js, no build step required.

### Tabs

| Tab | What It Shows |
|-----|---------------|
| **Home** | Model version, trade suggestions, cumulative P&L chart |
| **Pipeline** | Training progress (Markov + fine-tune), social media corpus stats + import, fine-tune loss chart, scheduled jobs, pipeline log |
| **Markets** | All Kalshi markets — active first, resolved last. Active at 1c/99c show as "Virtually Certain"/"Virtually Dead"; resolved show "Yes"/"No" |
| **Predictions** | Final blended predictions vs market prices, edge, component scores |
| **Trading** | Bot config (paper/live, auto-trade, Kelly fraction), portfolio, trade history |
| **System** | CPU/RAM/disk gauges, temperature, uptime, model version history, prediction accuracy |
| **TrumpGPT** | Interactive text generation — model selector (Markov Chain / Fine-Tuned Pythia), scenario, temperature, Q&A mode |

### Pipeline Tab Features

- **Pipeline Status**: Run/idle/error state, progress bar with elapsed/ETA, Monte Carlo simulation counter
- **Social Media Corpus**: Stats cards (tweets, Truth Social posts, daily digests, total words), Import Twitter Archive button with progress bar
- **Fine-Tune GPT-2/Pythia**: Epoch/loss/tokens-per-sec/ETA/RAM status cards, progress bar, live loss chart (Chart.js, polled every 15s), Start/Stop buttons
- **Scheduled Jobs**: Table of all 12 scheduler jobs with intervals
- **Pipeline Log**: Timestamped event log (max 200 entries)

### TrumpGPT Tab Features

- **Model selector**: Choose between Markov Chain (fast, ~instant) and Fine-Tuned Pythia (better quality, slower on CPU)
- **Scenario types**: Rally, Press Conference, Chopper Talk, Fox Interview, Social Media
- **Controls**: Word count, temperature (0.3-2.0), Q&A mode toggle
- **Output**: Generated text with word count

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
**Model:** `GET /model/status`, `GET /model/versions`, `GET /model/accuracy`
**Colab:** `GET /colab/predictions`, `POST /colab/import`, `POST /colab/save-to-db`, `GET /colab/discovered-phrases`
**Pipeline:** `GET /pipeline/status`, `GET /pipeline/training-status`, `GET /pipeline/log`, `POST /pipeline/run`, `POST /pipeline/export-upload`, `POST /pipeline/trigger-training`, `POST /pipeline/poll`
**Social Media:** `POST /social-media/import-twitter`, `POST /social-media/scrape-truth`, `GET /social-media/import-status`, `GET /social-media/stats`, `GET /social-media/recent-posts`
**Pythia:** `GET /fine-tune/pythia-status`, `POST /fine-tune/upload-predictions`
**TrumpGPT:** `POST /trumpgpt/generate`
**Drive:** `GET /drive/status`, `POST /drive/upload`, `POST /drive/download-predictions` (disabled in local mode)
**Live Monitor:** `POST /live/start`, `POST /live/stop`, `GET /live/status`
**System:** `POST /system/full-refresh`, `GET /system/health`, `GET /system/hardware`
**Alerts:** `GET /alerts`, `GET /alerts/count`, `POST /alerts/{id}/read`
**Config:** `GET /config/status`

## Key Dependencies

- **Web framework:** FastAPI + uvicorn
- **GUI:** Streamlit + Plotly (optional, legacy), Alpine.js + Chart.js (primary dashboard)
- **Database:** SQLAlchemy (SQLite)
- **Scraping:** BeautifulSoup4, requests, feedparser, trafilatura, yt-dlp, youtube-transcript-api, Playwright/Selenium
- **ML (local):** scikit-learn (GBM, RF, LogReg), pandas, numpy
- **ML (fine-tuning, optional):** torch, transformers, peft (LoRA), datasets, accelerate
- **ML (Colab):** Unsloth, TRL, transformers, bitsandbytes, PEFT
- **NLP:** spaCy, NLTK
- **Gemini:** google-generativeai (for news enrichment)
- **Google Drive:** google-api-python-client, google-auth
- **Scheduling:** APScheduler
- **Kalshi:** requests + RSA signing (cryptography)

## Data Flow Summary

```
┌─── RASPBERRY PI ───────────────────────────────────────────┐
│                                                             │
│  [10 scrapers + Truth Social + Twitter/X]                  │
│                    ↓                                        │
│              Speech table                                   │
│                    ↓                                        │
│        TermAnalyzer → TermOccurrence table                 │
│                    ↓                                        │
│            Markov Chain (Phase 1-5, ~35s)                   │
│                    ↓                                        │
│            Monte Carlo (2000 sims)                          │
│                    ↓                                        │
│        predictions_latest.json ←── blend if available ──┐  │
│                    ↓                                     │  │
│          TermPrediction table                            │  │
│                    ↓                                     │  │
│        TermPredictor (6-signal ensemble)                 │  │
│                    ↓                                     │  │
│        TradingBot (Kelly criterion)                      │  │
│                    ↓                                     │  │
│          Kalshi API → Trade table                        │  │
└──────────────────────────────────────────────────────────┘  │
                                                              │
┌─── MAC (fine-tuning) ──────────────────────────────────┐   │
│                                                         │   │
│  scp DB from Pi → fine_tune_mac.py                     │   │
│    → Pythia-410M LoRA fine-tune                        │   │
│    → Monte Carlo (200 sims)                            │   │
│    → POST predictions_pythia.json to Pi API ───────────┼───┘
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Known Data Quality Issues

- Many speeches in the DB have `date = 2026-03-08T20:34:*` — these are actually older speeches where `_extract_date_from_text()` failed and defaulted to `datetime.now()` during a batch scrape. The real dates should be re-extracted from titles or source pages.
- WhiteHouse.gov source was returning page chrome (100-word stubs like "Video Library") instead of actual transcripts. The scraper URL has been fixed to `/briefing-room/speeches-remarks/`.
- Roll Call/Factbase source returns news articles *about* Trump, not transcripts of his speech. These pollute the training corpus if not filtered.
- `data_exporter.py` `_chunk_transcript()` had an infinite loop bug when overlap settings caused the index to not advance — this has been fixed with a safety guard.

## Conventions

- All database access goes through `get_session()` context manager from `src/database/db.py`
- Scrapers return `int` (count of new speeches saved) and use `_save_speech()` for dedup (includes cross-source duplicate transcript detection)
- Config is a singleton at `src/config.py` — access via `from .config import config`
- Logging uses `loguru` (`from loguru import logger`) everywhere
- Tests use `unittest` — run with `python -m pytest tests/`
- Async background tasks in the API use FastAPI's `BackgroundTasks`
- Scheduler jobs are synchronous functions called by APScheduler's BackgroundScheduler
- Fine-tuning dependencies (torch, transformers, peft) are lazy-imported — the module can be imported on systems without these packages
- LoRA target modules are auto-detected from the model architecture — changing `FINE_TUNE_MODEL` to any HuggingFace causal LM works without code changes

## Probability Compression Fix (RESOLVED)

### Original Issue

When using Colab's LLM Monte Carlo, simulations generated ~430-word snippets but real Trump speeches are 5,000-10,000+ words. This compressed all probabilities toward ~40%.

### Fixes Applied

1. **Fix A: Poisson length normalization** — Implemented in `colab_integration.py:_apply_poisson_correction()`. Reads `avg_mentions_per_speech` from predictions JSON and corrects to full speech length using `P = 1 - exp(-lambda * target_words)`. When using local Markov chain (which generates full-length speeches per scenario), the correction is skipped automatically.

2. **Fix G: Per-scenario length normalization** — The local Markov chain generates scenario-appropriate lengths (rally=5000w, press_conference=2000w, etc.), so probabilities are naturally correct for each scenario type.

3. **Dynamic snippet detection** — `_apply_poisson_correction` reads `simulation_params.avg_words_per_speech` from the predictions JSON. If simulations are already near rally length (>=80%), Poisson correction is skipped.

4. **Pythia Monte Carlo** — The Pythia fine-tuner generates shorter sims (500 words) for speed and relies on the existing Poisson correction infrastructure. These are blended 40% with the Markov predictions (which use full-length sims and don't need correction).

### Remaining Potential Improvements

- ~~**Fix C**: Gate temporal/trend on data sufficiency~~ → **DONE** (1B: temporal gated when dates poisoned; 1D: confidence threshold lowered)
- ~~**Fix D**: Platt scaling calibration using settled market outcomes~~ → **DONE** (4B: `scripts/backfill_settlements.py` with optional Platt scaling)
- **Fix F**: Market price as Bayesian prior instead of arithmetic averaging
