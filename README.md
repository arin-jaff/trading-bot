# TrumpGPT Trading Bot

Fully automated trading system that predicts which words/phrases Donald Trump will say in upcoming speeches, trades on those predictions via [Kalshi](https://kalshi.com) prediction markets, and continuously improves its model by scraping new speech data.

Designed to run autonomously on a Raspberry Pi 4 — plug in, fund it, walk away.

## How It Works

```
┌─────────────────────── RASPBERRY PI 4 ───────────────────────────┐
│                                                                   │
│  Every 6 hours:                                                  │
│    Scrape speeches ──► Train Markov chain ──► Monte Carlo sims   │
│    ──► Term probability predictions ──► Kelly criterion trading  │
│                                                                   │
│  Optional (FINE_TUNE_ENABLED=true):                              │
│    Fine-tune Pythia-410M with LoRA ──► Blend predictions         │
│    (~9 hours, runs at lowest CPU priority in background)         │
│                                                                   │
│  Continuous:                                                     │
│    Market sync (5min) • News enrichment (1hr)                    │
│    Prediction refresh (15min) • Trade check (5min)               │
│    Event tracking (30min) • Live speech monitor (1min)           │
│                                                                   │
│  API + Dashboard: FastAPI :8000                                  │
│  Database: SQLite ─────── Email: SMTP notifications              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──── Kalshi ────────┐
                    │  RSA key-pair auth │
                    │  Market data       │
                    │  Order execution   │
                    └────────────────────┘
```

### The Core Loop

1. **Scrape** — 10 sources (White House, Rev.com, C-SPAN, YouTube, etc.) collect Trump speech transcripts. Truth Social posts scraped every 2 hours and grouped into daily digests. Twitter archive (~56K tweets) auto-imported on first run.
2. **Train** — An order-3 word-level Markov chain learns Trump's speech patterns from the corpus (~5 seconds)
3. **Simulate** — 2,000 Monte Carlo simulated speeches across 5 scenario types (rally, press conference, chopper talk, interview, social media)
4. **Predict** — Count term occurrences across simulations, blend with 5 other signals (historical frequency, temporal patterns, trends, event correlation, news relevance) into final probabilities
5. **Trade** — Compare predictions to Kalshi market prices, execute trades where our edge exceeds the threshold using Kelly criterion position sizing
6. **(Optional) Fine-tune** — Train Pythia-410M (EleutherAI) with LoRA on the Pi 4 CPU, blend its Monte Carlo predictions with Markov results for higher accuracy

## Directory Structure

```
src/
  config.py                    # Central config from env vars
  scheduler.py                 # APScheduler — 11+ background jobs
  alerts.py                    # Alert manager + desktop/email notifications

  api/
    server.py                  # FastAPI backend — 50+ endpoints

  gui/
    dashboard.py               # Streamlit dashboard (legacy, optional)

  database/
    db.py                      # SQLAlchemy engine, session management
    models.py                  # 10 ORM models (Market, Term, Speech, Trade, ModelVersion, etc.)

  kalshi/
    client.py                  # KalshiClient — RSA auth, rate limiting, REST API
    market_sync.py             # Syncs markets, extracts terms from titles
    trading_bot.py             # Kelly criterion, risk limits, position management

  scraper/
    speech_scraper.py          # 10 scraping sources
    social_media_importer.py   # Bulk Twitter archive + Truth Social import
    term_analyzer.py           # Extracts term occurrences from transcripts
    event_tracker.py           # Discovers upcoming Trump events
    live_monitor.py            # Real-time term detection during live speeches

  ml/
    markov_trainer.py          # Markov chain trainer + Monte Carlo simulator
    fine_tuner.py              # Pi-native Pythia-410M LoRA fine-tuning
    local_pipeline.py          # 8-phase local training pipeline
    predictor.py               # 6-signal weighted ensemble predictor
    colab_integration.py       # Loads predictions + Poisson correction
    feature_engineering.py     # 30+ features from speech data
    model_trainer.py           # GBM, RF, LogReg ensemble (fallback)
    data_exporter.py           # Exports corpus for training
    news_enrichment.py         # Gemini 2.0 Flash Lite current events
    colab_pipeline.py          # Colab+Drive pipeline (dormant in local mode)
    drive_sync.py              # Google Drive upload/download (dormant in local mode)

  notifications/
    email_notifier.py          # Trade alerts, daily digest, critical alerts via SMTP

static/
  index.html                   # Single-page dashboard (Alpine.js + Chart.js)

deploy/
  setup-pi.sh                 # One-command Raspberry Pi setup script
  trumpbot-api.service         # systemd service for API + scheduler
  trumpbot-gui.service         # systemd service for Streamlit dashboard
  watchdog.sh                  # Health check cron (restarts on failure)

data/
  exports/                     # Training data exports
  imports/                     # Downloaded social media archives
  models/                      # Markov pickles + Pythia LoRA adapters
  predictions/                 # predictions_latest.json + timestamped copies
```

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd trading-bot

# Full installation (development machine)
make install

# OR lightweight Pi installation (no torch/transformers)
make install-pi
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (see Environment Variables below)
```

### 3. Initialize Database

```bash
make init
```

### 4. Start the Bot

```bash
# Start API server + scheduler + dashboard (all-in-one)
make api
```

Open **http://localhost:8000** in your browser to see the dashboard.

The scheduler starts automatically and handles:
- Market syncing every 5 minutes
- Speech scraping every 2 hours
- Markov training pipeline every 6 hours
- Prediction generation every 15 minutes
- Trading checks every 5 minutes
- And 7 more background jobs

### 5. (Optional) Expand the Corpus

Import ~56K Trump tweets to massively expand the training data:

```bash
make import-twitter
```

Or from the dashboard: **Pipeline** tab → **Import Twitter Archive** button.

### 6. (Optional) Enable Fine-Tuning

For higher-quality text generation using Pythia-410M:

```bash
# Install PyTorch + transformers + PEFT
make install-finetune

# Enable in .env
echo "FINE_TUNE_ENABLED=true" >> .env

# Restart the API
make api
```

Fine-tuning auto-triggers when the corpus grows by 50+ speeches since the last run. It runs as Phase 6-8 in a background thread at lowest CPU priority. You can also trigger it manually from the dashboard: **Pipeline** tab → **Start Fine-Tuning**.

## Raspberry Pi Deployment

### Prerequisites
- Raspberry Pi 4 (8GB RAM recommended for fine-tuning, 2GB+ for Markov-only)
- Raspberry Pi OS Lite 64-bit
- 128GB+ MicroSD card
- SSH access + internet connection

### Setup

```bash
# On the Pi:
cd /home/pi
git clone <repo-url>
cd trading-bot

# Copy secrets from your development machine:
# scp .env pi@<pi-ip>:/home/pi/trading-bot/.env
# scp -r secrets/ pi@<pi-ip>:/home/pi/trading-bot/secrets/

# Run the setup script (installs everything, creates services)
make deploy-pi

# Start
sudo systemctl start trumpbot-api
sudo systemctl start trumpbot-gui   # optional — saves RAM if skipped

# Verify
curl http://localhost:8000/api/system/health
```

### What the Setup Script Does
1. Installs system packages (python3-venv, libffi, libssl, etc.)
2. Creates Python virtual environment
3. Installs lightweight dependencies (`requirements-pi.txt` — no torch/transformers)
4. Downloads NLTK data
5. Initializes SQLite database
6. Installs systemd services (auto-start on boot, auto-restart on crash)
7. Sets up watchdog cron (checks health every 5 min, restarts if down)

### Enabling Fine-Tuning on the Pi

```bash
# SSH into the Pi, then:
cd /home/pi/trading-bot
source venv/bin/activate

# Install fine-tuning deps (~1.5GB download)
make install-finetune

# Add to .env
echo "FINE_TUNE_ENABLED=true" >> .env

# Restart the service
sudo systemctl restart trumpbot-api
```

Fine-tuning uses ~3GB RAM (out of 8GB) and runs at lowest CPU priority (`os.nice(19)`), so the bot continues trading normally. Training takes ~9 hours for 3 epochs — it runs overnight.

### Access
- **Dashboard**: `http://<pi-ip>:8000`
- **API**: `http://<pi-ip>:8000/api/`
- **Logs**: `journalctl -u trumpbot-api -f`

## Environment Variables

```bash
# Kalshi API (required for trading)
KALSHI_API_KEY=                        # API key from kalshi.com
KALSHI_PRIVATE_KEY_PATH=secrets/kalshi_private_key.pem

# Email Notifications
EMAIL_ENABLED=true
EMAIL_SMTP_HOST=smtp.gmail.com         # Works with Google Workspace
EMAIL_SMTP_PORT=587
EMAIL_FROM=you@example.com
EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx # App password from Google
EMAIL_TO=you@example.com

# Pipeline Mode
PIPELINE_MODE=local                    # 'local' (Pi) or 'colab' (GPU cloud)
MARKOV_ORDER=3                         # Markov chain n-gram order
MONTE_CARLO_SIMULATIONS=2000           # Number of simulated speeches
RETRAIN_INTERVAL_HOURS=6               # How often to retrain

# Fine-tuning (optional)
FINE_TUNE_ENABLED=false                # Set to 'true' to enable Pythia-410M LoRA
FINE_TUNE_MODEL=EleutherAI/pythia-410m # Any HuggingFace causal LM (auto-detects LoRA targets)
FINE_TUNE_EPOCHS=3                     # ~9 hours for 3 epochs on Pi 4
FINE_TUNE_LORA_RANK=16                 # LoRA rank (16 = ~3GB peak RAM)
FINE_TUNE_LR=5e-4                      # Learning rate
FINE_TUNE_MC_SIMS=200                  # Monte Carlo sims (fewer than Markov — slower per sim)

# Google Drive (only needed if PIPELINE_MODE=colab)
GOOGLE_DRIVE_FOLDER_ID=
GOOGLE_SERVICE_ACCOUNT_KEY_PATH=

# Optional
YOUTUBE_API_KEY=                       # For YouTube transcript scraper
GEMINI_API_KEY=                        # For current events enrichment
DATABASE_URL=sqlite:///data/trading_bot.db
LOG_LEVEL=INFO
REFRESH_INTERVAL_SECONDS=60
API_PORT=8000
GUI_PORT=8501
```

## Dashboard

The primary dashboard is a single-page app served at `http://localhost:8000`. Built with Alpine.js + Chart.js.

| Tab | What It Shows |
|-----|---------------|
| **Home** | Model version, trade suggestions, cumulative P&L chart |
| **Pipeline** | Training progress, social media corpus import, fine-tune status + loss chart, scheduled jobs, log |
| **Markets** | All Kalshi markets — active first, resolved last. 1c/99c show "Virtually Certain"/"Virtually Dead"; resolved show "Yes"/"No" |
| **Predictions** | Final blended predictions vs market prices, edge, component scores |
| **Trading** | Bot config (paper/live, auto-trade, Kelly fraction), portfolio, trade history |
| **System** | CPU/RAM/disk gauges, temperature, uptime, model version history, accuracy |
| **TrumpGPT** | Interactive text generation with model selector (Markov / Pythia), scenario, temperature, Q&A mode |

## Prediction Engine

### Ensemble Weights

| Signal | Weight | Source |
|--------|--------|--------|
| Monte Carlo | 0.40 | Markov chain simulated speeches (2,000 sims) |
| Frequency | 0.20 | Recency-weighted historical speech frequency (60-day half-life) |
| Trend | 0.15 | Recent usage velocity and acceleration |
| Event Correlation | 0.10 | How event type correlates with term usage |
| News Relevance | 0.10 | Gemini 2.0 Flash Lite current events talking points |
| Temporal | 0.05 | Day-of-week and seasonal patterns |

### Training Pipeline (Pi Mode)

Every 6 hours (or when 5+ new speeches are scraped):

| Phase | What | Time |
|-------|------|------|
| 0 | Social media refresh (auto-import Twitter + scrape Truth Social + rebuild digests) | ~10s |
| 1 | Train Markov chain (order-3 on all transcripts) | ~5s |
| 2 | Load tracked terms from DB | <1s |
| 3 | Monte Carlo simulation (2,000 speeches x 5 scenarios) | ~30s |
| 4 | Save predictions JSON | <1s |
| 5 | Import to database + create ModelVersion | <1s |
| 6 | Fine-tune Pythia-410M with LoRA | ~9h (background) |
| 7 | Pythia Monte Carlo (200 sims) | ~hours (background) |
| 8 | Blend Markov + Pythia predictions (60/40) | <1s |

Phases 1-5 complete in ~35 seconds. Phases 6-8 auto-trigger in a background thread when `FINE_TUNE_ENABLED=true` AND the corpus has grown by 50+ speeches since the last fine-tune. No manual scheduling needed.

## Trading Bot

### Risk Management
- **Kelly Criterion** position sizing (half-Kelly for safety)
- **Max position**: 100 contracts per market
- **Daily loss limit**: Configurable (default $50), hard stop
- **Cooldown**: 2-hour trading pause at 50% of daily loss limit
- **Drawdown protection**: Halts trading if balance drops 30% from peak
- **Edge threshold**: Only trades when predicted probability differs 5%+ from market price
- **Confidence gate**: Only trades on predictions with 30%+ confidence

### Market Status Display
- **Active markets at 1c**: shown as **"Virtually Dead"** — the term almost certainly won't be said. Untradeable.
- **Active markets at 99c**: shown as **"Virtually Certain"** — the term was already said but the market hasn't settled yet. Untradeable.
- **Resolved markets**: shown as **"Yes"** or **"No"** based on final result.
- Active markets at 1c/99c are automatically skipped by the trading bot.

### Going Live
```bash
curl -X PUT http://<pi-ip>:8000/api/trading/config \
  -H "Content-Type: application/json" \
  -d '{
    "auto_trade": true,
    "paper_mode": false,
    "max_daily_loss": 10,
    "max_total_exposure": 100,
    "kelly_fraction": 0.25
  }'
```

## API Endpoints

### Core
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/markets` | All tracked markets |
| POST | `/api/markets/sync` | Trigger Kalshi market sync |
| GET | `/api/terms` | All tracked terms with stats |
| POST | `/api/speeches/scrape` | Scrape all 10 sources |
| GET | `/api/predictions/final` | Final blended predictions vs market prices |
| GET | `/api/trading/suggestions` | Current trade suggestions |
| PUT | `/api/trading/config` | Update bot config (paper_mode, auto_trade, etc.) |
| POST | `/api/trading/execute` | Execute a trade |
| GET | `/api/system/health` | Health check |
| GET | `/api/system/hardware` | CPU, RAM, disk, temperature |

### Pipeline & Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/pipeline/run` | Trigger full training pipeline |
| GET | `/api/pipeline/training-status` | Real-time training progress |
| GET | `/api/pipeline/log` | Pipeline event log |
| POST | `/api/fine-tune/start` | Start Pythia fine-tuning |
| POST | `/api/fine-tune/stop` | Graceful stop (saves checkpoint) |
| GET | `/api/fine-tune/status` | Epoch, loss, ETA, tokens/sec, RAM |
| GET | `/api/fine-tune/loss-history` | Loss curve data for charting |

### Social Media Import
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/social-media/import-twitter` | Download + import Trump Twitter archive |
| POST | `/api/social-media/import-truth` | Import Truth Social JSON dump |
| GET | `/api/social-media/import-status` | Import progress |
| GET | `/api/social-media/stats` | Corpus stats (tweets, Truth Social, digests) |

### Text Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/trumpgpt/generate` | Generate text (model: 'markov' or 'gpt2') |

### Model & History
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/model/status` | Active model info, version, predictions |
| GET | `/api/model/versions` | All model version records |
| GET | `/api/model/accuracy` | Prediction accuracy vs settled markets |
| GET | `/api/trades/history` | Paginated trade history with P&L |

## Scheduled Jobs

| Job | Interval | What It Does |
|-----|----------|-------------|
| Market sync | 5 min | Syncs Kalshi market data, alerts on new markets |
| Speech scrape | 2 hours | Scrapes all 10 sources, analyzes terms |
| Social media | 2 hours | Scrapes Truth Social + rebuilds daily digests |
| Event tracking | 30 min | Discovers upcoming Trump events |
| Live check | 1 min | Checks if any events are currently live |
| Predictions | 15 min | Generates ensemble predictions |
| Trading check | 5 min | Checks suggestions, auto-trades if enabled |
| News enrichment | 1 hour | Gemini current events talking points |
| Position mgmt | 5 min | Profit-taking and stop-loss checks |
| Local pipeline | 6 hours | Train Markov chain + Monte Carlo + import |
| Daily digest | 8 AM UTC | Sends email summary |
| Fine-tune | Auto | Pythia-410M LoRA training — auto-triggers when corpus grows 50+ speeches |

## Make Targets

```bash
make install          # pip install + spacy model
make install-pi       # Lightweight Pi dependencies
make install-finetune # torch + transformers + peft + datasets + accelerate
make init             # Initialize database
make api              # Start FastAPI server + scheduler + dashboard
make gui              # Start Streamlit dashboard (optional)
make all              # Start both
make import-twitter   # Download + import Trump Twitter archive (~56K tweets)
make deploy-pi        # Run Pi setup script
make export-colab     # Export training data for Colab
make clean            # Delete DB + caches
```

## Model Versioning

Markov chain versions start at `1.0.X`, Pythia fine-tune versions at `2.0.X`:

```
TrumpGPT v1.0.0  — markov_chain  — 127 speeches  — 5.2s training  — 2000 sims
TrumpGPT v1.0.1  — markov_chain  — 132 speeches  — 5.4s training  — 2000 sims
TrumpGPT v2.0.0  — gpt2_lora     — 2891 speeches — 32400s training — 200 sims
```

Versions auto-increment the patch number. Markov artifacts are pickle files in `data/models/`. Pythia adapters are in `data/models/gpt2_lora/adapter_latest/` (~2-3MB).

## Known Issues

- Many speeches in the DB have incorrect dates (2026-03-08) from failed date extraction during batch scraping
- Roll Call/Factbase source sometimes returns news articles about Trump rather than actual transcripts
- Some non-YouTube scrapers return 0 speeches when HTML selectors become outdated
- Probability compression fix (Poisson normalization) is applied but may over-correct for very short terms
- Fine-tuning on Pi 4 takes ~9 hours per run — this is expected for CPU-only training of a 410M parameter model
