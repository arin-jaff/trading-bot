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
│  Continuous:                                                     │
│    Market sync (5min) • Arbitrage scan (10min)                   │
│    Prediction refresh (15min) • Trade check (5min)               │
│    Event tracking (30min) • Live speech monitor (1min)           │
│                                                                   │
│  API: FastAPI :8000 ──── Dashboard: Streamlit :8501              │
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

1. **Scrape** — 10 sources (White House, Rev.com, C-SPAN, YouTube, etc.) collect Trump speech transcripts
2. **Train** — An order-3 word-level Markov chain learns Trump's speech patterns from the corpus (~5 seconds)
3. **Simulate** — 2,000 Monte Carlo simulated speeches across 5 scenario types (rally, press conference, chopper talk, interview, social media)
4. **Predict** — Count term occurrences across simulations, blend with 4 other signals (historical frequency, temporal patterns, trends, event correlation) into final probabilities
5. **Trade** — Compare predictions to Kalshi market prices, execute trades where our edge exceeds the threshold using Kelly criterion position sizing

## Directory Structure

```
src/
  config.py                    # Central config from env vars
  scheduler.py                 # APScheduler — 10+ background jobs
  alerts.py                    # Alert manager + desktop/email notifications

  api/
    server.py                  # FastAPI backend — 40+ endpoints

  gui/
    dashboard.py               # Streamlit dashboard — 11 tabs

  database/
    db.py                      # SQLAlchemy engine, session management
    models.py                  # 10 ORM models (Market, Term, Speech, Trade, ModelVersion, etc.)

  kalshi/
    client.py                  # KalshiClient — RSA auth, rate limiting, REST API
    market_sync.py             # Syncs markets, extracts terms from titles
    trading_bot.py             # Kelly criterion, risk limits, arbitrage scanner

  scraper/
    speech_scraper.py          # 10 scraping sources
    term_analyzer.py           # Extracts term occurrences from transcripts
    event_tracker.py           # Discovers upcoming Trump events
    live_monitor.py            # Real-time term detection during live speeches

  ml/
    markov_trainer.py          # Markov chain trainer + Monte Carlo simulator
    local_pipeline.py          # Local training pipeline (Pi mode)
    predictor.py               # 5-signal weighted ensemble predictor
    colab_integration.py       # Loads predictions + Poisson correction
    feature_engineering.py     # 30+ features from speech data
    model_trainer.py           # GBM, RF, LogReg ensemble (fallback)
    data_exporter.py           # Exports corpus for training
    colab_pipeline.py          # Colab+Drive pipeline (dormant in local mode)
    drive_sync.py              # Google Drive upload/download (dormant in local mode)

  notifications/
    email_notifier.py          # Trade alerts, daily digest, critical alerts via SMTP

deploy/
  setup-pi.sh                 # One-command Raspberry Pi setup script
  trumpbot-api.service         # systemd service for API + scheduler
  trumpbot-gui.service         # systemd service for Streamlit dashboard
  watchdog.sh                  # Health check cron (restarts on failure)

notebooks/
  01_finetune_trump_llm.ipynb  # LoRA fine-tuning (Colab mode only)
  02_monte_carlo_predictor.ipynb # Monte Carlo simulation (Colab mode only)

data/
  exports/                     # Training data exports
  models/                      # Markov chain pickle files (markov_v1.0.0.pkl, etc.)
  predictions/                 # predictions_latest.json + timestamped copies
```

## Quick Start (Local Development)

```bash
# 1. Clone
git clone https://github.com/arin-jaff/trading-bot.git
cd trading-bot

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env   # Edit with your API keys

# 4. Initialize database
make init

# 5. Start API + scheduler
make api

# 6. Start dashboard (separate terminal)
make gui

# 7. Open http://localhost:8501
```

## Raspberry Pi Deployment

### Prerequisites
- Raspberry Pi 4 (2GB+ RAM) with Raspberry Pi OS Lite 64-bit
- SSH access enabled
- Internet connection

### Setup

```bash
# On your Mac: push code to GitHub, then on the Pi:
cd /home/pi
git clone https://github.com/arin-jaff/trading-bot.git
cd trading-bot

# Copy secrets from your Mac:
# scp .env pi@<pi-ip>:/home/pi/trading-bot/.env
# scp -r secrets/ pi@<pi-ip>:/home/pi/trading-bot/secrets/

# Run the setup script (installs everything, creates services)
bash deploy/setup-pi.sh

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

### Access
- **Dashboard**: `http://<pi-ip>:8501`
- **API**: `http://<pi-ip>:8000`
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

## Dashboard Tabs

| Tab | What It Shows |
|-----|---------------|
| **Home** | Trade suggestion cards (accept/deny), active model version |
| **Markets** | All Kalshi Trump Mentions markets with prices, volume, status |
| **Terms** | Tracked terms with occurrence counts and trend scores |
| **Trading** | Bot config, suggestions, portfolio summary |
| **Trade History** | All past trades with P&L, cumulative P&L chart, win rate |
| **Events** | Upcoming Trump events, live status |
| **Live Monitor** | Real-time term detection during active speeches |
| **TrumpGPT** | Model status, training progress bar with elapsed/ETA, predictions |
| **Model Versions** | Version history (TrumpGPT v1.0.0, v1.0.1...) with corpus size, timing |
| **Pi Status** | CPU/RAM/disk gauge charts, temperature, uptime, load averages |
| **Data** | Speech collection stats, term frequency report, system health |

## Prediction Engine

### Ensemble Weights

| Signal | Weight | Source |
|--------|--------|--------|
| Monte Carlo | 0.40 | Markov chain simulated speeches (2,000 sims) |
| Frequency | 0.20 | Recency-weighted historical speech frequency (60-day half-life) |
| Event Correlation | 0.15 | How event type correlates with term usage |
| Trend | 0.15 | Recent usage velocity and acceleration |
| Temporal | 0.10 | Day-of-week and seasonal patterns |

### Local Training Pipeline (Pi Mode)

Every 6 hours (or when 5+ new speeches are scraped):

1. **Train Markov Chain** (~5s) — Order-3 word-level chain on all processed transcripts
2. **Monte Carlo Simulation** (~100s) — Generate 2,000 speeches across 5 scenario types
3. **Count Terms** — For each tracked term, compute P(appears in speech)
4. **Poisson Correction** — Adjust probabilities for speech length differences
5. **Save to DB** — Import predictions for the ensemble predictor to use
6. **Version Model** — Create ModelVersion record (TrumpGPT v1.0.X)

### Colab Pipeline (GPU Mode — Optional)

If `PIPELINE_MODE=colab`, the bot instead:
1. Exports training data to Google Drive
2. Triggers a Colab notebook that LoRA fine-tunes Llama-3.1-8B
3. Runs 1,000 Monte Carlo simulations on the fine-tuned model
4. Polls Drive for results and imports predictions

To switch: set `PIPELINE_MODE=colab` in `.env` and restart.

## Trading Bot

### Risk Management
- **Kelly Criterion** position sizing (half-Kelly for safety)
- **Max position**: 100 contracts per market
- **Daily loss limit**: Configurable (default $50), hard stop
- **Cooldown**: 2-hour trading pause at 50% of daily loss limit
- **Drawdown protection**: Halts trading if balance drops 30% from peak
- **Edge threshold**: Only trades when predicted probability differs 5%+ from market price
- **Confidence gate**: Only trades on predictions with 30%+ confidence

### Arbitrage Scanner
Scans every 10 minutes for:
- **Spread arbitrage**: YES + NO price < $0.98 (guaranteed profit minus fees)
- **Overpriced spreads**: YES + NO price > $1.02 (sell both sides)

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

## Email Notifications

| Email | When | Content |
|-------|------|---------|
| **Trade Alert** | On each trade execution | Market, side, quantity, price, edge |
| **Daily Digest** | 8 AM UTC daily | P&L summary, trades, top predictions, scraper health |
| **Critical Alert** | Loss limit hit, training failure, drawdown | Details + auto-paused status |

## API Endpoints

### Markets
- `GET /api/markets` — All tracked markets
- `GET /api/markets/weekly-payouts` — Settlement history by week
- `POST /api/markets/sync` — Trigger Kalshi market sync

### Terms
- `GET /api/terms` — All tracked terms with stats
- `GET /api/terms/{id}/history` — Term usage time series
- `GET /api/terms/report` — Full frequency report

### Speeches
- `POST /api/speeches/scrape` — Trigger scraping from all 10 sources
- `GET /api/speeches/stats` — Collection statistics

### Events
- `GET /api/events` — Upcoming Trump events
- `GET /api/events/live` — Currently live events
- `POST /api/events/update` — Trigger event discovery

### Predictions
- `GET /api/predictions` — Latest ensemble predictions
- `POST /api/predictions/generate` — Trigger prediction generation
- `GET /api/predictions/final` — Final blended predictions vs market prices

### Trading
- `GET /api/trading/suggestions` — Current trade suggestions
- `GET /api/trading/portfolio` — Portfolio summary
- `GET /api/trading/config` — Bot configuration
- `PUT /api/trading/config` — Update bot configuration
- `POST /api/trading/execute` — Execute a trade

### Pipeline
- `GET /api/pipeline/status` — Pipeline status + mode
- `GET /api/pipeline/training-status` — Real-time training progress (elapsed, ETA, stage)
- `POST /api/pipeline/run` — Trigger training pipeline

### Model
- `GET /api/model/status` — Active model info, version, method, predictions
- `GET /api/model/versions` — All model version records

### System
- `GET /api/system/health` — Health check
- `GET /api/system/hardware` — CPU%, RAM%, disk%, temperature, uptime
- `POST /api/system/full-refresh` — Full data refresh (sync + scrape + predict)

### Trades
- `GET /api/trades/history` — Paginated trade history with P&L summary

### Alerts
- `GET /api/alerts` — Recent alerts (filterable)
- `GET /api/alerts/count` — Unread count
- `POST /api/alerts/{id}/read` — Mark as read

### Drive/Colab (disabled in local mode)
- `GET /api/drive/status`
- `POST /api/drive/upload`
- `POST /api/drive/download-predictions`
- `POST /api/pipeline/export-upload`
- `POST /api/pipeline/trigger-training`
- `POST /api/pipeline/poll`

## Database Models

| Model | Purpose |
|-------|---------|
| **Market** | Kalshi market: ticker, prices, volume, status, result |
| **Term** | Tracked word/phrase: normalized form, compound flag, trend score |
| **Speech** | Scraped transcript: source, title, type, date, word count |
| **TermOccurrence** | Term+Speech join: count, context snippets |
| **TrumpEvent** | Upcoming appearance: type, location, time, topics |
| **TermPrediction** | ML prediction: probability, confidence, model name, features |
| **PriceSnapshot** | Historical market prices |
| **Trade** | Executed trade: side, quantity, price, P&L, strategy |
| **ModelVersion** | Model iteration: version string, corpus size, training duration, metrics |
| **BotConfig** | Persistent key/value config store |

## Speech Scraper Sources

| # | Source | Method |
|---|--------|--------|
| 1 | Rev.com | Human-verified transcripts |
| 2 | Google News RSS | Meta-source for transcript links |
| 3 | WhiteHouse.gov | Official remarks |
| 4 | Roll Call / Factbase | Speech archives |
| 5 | C-SPAN | Video metadata |
| 6 | C-SPAN Transcripts | Transcript extraction from video library |
| 7 | YouTube (API) | Data API + youtube-transcript-api |
| 8 | YouTube (yt-dlp) | Auto-subtitle extraction (no API key) |
| 9 | Presidency Project | UCSB American Presidency Project |
| 10 | Trump Twitter Archive | Historical tweets |

## Scheduled Jobs

| Job | Interval | What It Does |
|-----|----------|-------------|
| Market sync | 5 min | Syncs Kalshi market data |
| Speech scrape | 2 hours | Scrapes all 10 sources, analyzes terms |
| Event tracking | 30 min | Discovers upcoming Trump events |
| Live check | 1 min | Checks if any events are currently live |
| Predictions | 15 min | Generates ensemble predictions |
| Trading check | 5 min | Checks suggestions, auto-trades if enabled |
| Arbitrage scan | 10 min | Scans for spread mispricing |
| Local pipeline | 6 hours | Train Markov chain + Monte Carlo + import |
| Daily digest | 8 AM UTC | Sends email summary |

## Make Targets

```bash
make install      # pip install + spacy model
make install-pi   # Lightweight Pi dependencies
make init         # Initialize database
make api          # Start FastAPI server
make gui          # Start Streamlit dashboard
make all          # Start both
make deploy-pi    # Run Pi setup script
make export-colab # Export training data for Colab
make clean        # Delete DB + caches
```

## Model Versioning

Each training run creates a new model version:

```
TrumpGPT v1.0.0  — 2026-03-16 20:15  — 127 speeches  — 5.2s training  — 2000 sims
TrumpGPT v1.0.1  — 2026-03-17 02:15  — 132 speeches  — 5.4s training  — 2000 sims
TrumpGPT v1.0.2  — 2026-03-17 08:15  — 132 speeches  — skipped (no new data)
```

Versions auto-increment the patch number. Model artifacts are saved as pickle files in `data/models/`.

## Known Issues

- Many speeches in the DB have incorrect dates (2026-03-08) from failed date extraction during batch scraping
- Roll Call/Factbase source sometimes returns news articles about Trump rather than actual transcripts
- Some non-YouTube scrapers return 0 speeches when HTML selectors become outdated
- Probability compression fix (Poisson normalization) is applied but may over-correct for very short terms
