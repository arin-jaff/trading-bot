# Trump Mentions Trading Bot

Automated trading system for Kalshi Trump Mentions markets. Scrapes, analyzes, and predicts Trump's speech patterns to find profitable trading opportunities.

## Architecture

```
src/
├── kalshi/           # Kalshi API client, market sync, trading bot
├── scraper/          # Speech scraping, term analysis, event tracking, live monitoring
├── ml/               # Prediction engine, feature engineering, model training
├── database/         # SQLAlchemy models and session management
├── api/              # FastAPI backend server
├── gui/              # Streamlit dashboard
├── alerts.py         # Alert/notification system
├── config.py         # Configuration management
└── scheduler.py      # Background job scheduling
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Initialize database:**
   ```bash
   make init
   ```

4. **Run the system:**
   ```bash
   # Terminal 1: API server (with background scheduler)
   make api

   # Terminal 2: Dashboard
   make gui
   ```

5. **Open the dashboard:**
   Navigate to `http://localhost:8501`

## Features

- **Market Tracking**: Auto-discovers all Trump Mentions markets on Kalshi
- **Term Database**: Tracks single words, multi-word phrases, and compound terms (e.g., "X / Y")
- **Speech Scraping**: Sources from White House, Rev.com, Factba.se, C-SPAN, Miller Center, YouTube
- **ML Predictions**: Ensemble of gradient boosting, random forest, and logistic regression
- **LLM Analysis**: Contextual prediction using Claude/GPT
- **Live Monitoring**: Real-time detection during live Trump speeches
- **Trading Bot**: Kelly criterion sizing, risk limits, auto-trade capability
- **Event Calendar**: Upcoming Trump appearances with live alerts
- **Desktop Notifications**: macOS alerts for critical events

## API Keys Required

- **Kalshi**: Email/password for trading API
- **Anthropic or OpenAI**: For LLM-enhanced predictions (optional)
- **YouTube Data API**: For video/transcript scraping (optional)
