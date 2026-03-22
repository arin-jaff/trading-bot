# Pipeline & Training Sources

## Data Sources (as of March 2026)

| # | Source | Status | What It Gets | Volume |
|---|--------|--------|-------------|--------|
| 1 | Rev.com | **Working** | Human-verified transcripts | ~18 per scrape |
| 2 | Google News RSS | **Working** | Aggregates transcript links across outlets (also covers C-SPAN) | ~61 matches per scrape |
| 3 | White House `/remarks/` | **Working** (fixed) | Official remarks + video pages | ~4/page x 88 pages |
| 4 | Roll Call/Factbase | **Disabled** | Was returning news articles, not transcripts — polluted corpus | — |
| 5 | C-SPAN search | **Disabled** | JS-rendered, no server-side content. Covered by #2 | — |
| 6 | C-SPAN transcripts | **Disabled** | Same JS issue. Covered by #2 + #8 | — |
| 7 | YouTube Data API | **Inactive** | Needs `YOUTUBE_API_KEY` | — |
| 8 | YouTube yt-dlp | **Working** (rewritten) | White House + RSBN channels via `youtube-transcript-api` | 100 WH + 50 RSBN videos |
| 9 | Presidency Project (UCSB) | **Working** (fixed) | Gold-standard presidential documents | 25/page x 50 pages |
| 10 | Twitter Archive | **Disabled** | JS-only shell. Covered by bulk importer | — |
| 11 | Truth Social | **Working** | Live scraping via Mastodon API (**every 30 min**) | Ongoing |
| 12 | Twitter/X (Nitter) | **Working** | Live scraping via Nitter RSS proxies (**every 30 min**) | Ongoing |
| 13 | Bulk Tweet Import | **Working** | ~56K historical tweets, auto-imported on first run | One-time |

**Active transcript sources:** Rev.com, Google News RSS, White House, YouTube yt-dlp, Presidency Project (5 sources)
**Active social media (PRIMARY):** Truth Social, Twitter/X, bulk tweet archive (3 sources, scraped every 30 min)

## Architecture

Everything runs on the Raspberry Pi. No Mac, no Colab, no external compute.

```
┌─────────────────── RASPBERRY PI 4 ──────────────────────────────┐
│                                                                   │
│  Scheduler (APScheduler, 14+ jobs)                               │
│    • Social media scrape ───── every 30 min (Truth Social + X)   │
│    • Social trend analysis ─── every 30 min (TF-IDF + velocity)  │
│    • Market sync ────────────── every 5 min (Kalshi API)         │
│    • Speech scraping ────────── every 2 hours (10 sources)       │
│    • Predictions ────────────── every 15 min (7-signal ensemble) │
│    • Trading checks ─────────── every 5 min (fee-aware Kelly)    │
│    • Position management ────── every 5 min (profit-take/stop)   │
│    • News enrichment ────────── every 1 hour (Gemini Flash Lite) │
│    • Local pipeline ─────────── every 6 hours (Markov + MC)      │
│    • Pi fine-tuning ─────────── 2 AM ET nightly (Pythia LoRA)    │
│    • Event tracking ─────────── every 30 min                     │
│    • Daily email digest ─────── 8 AM ET                          │
│                                                                   │
│  API: FastAPI on :8000 ──── Dashboard: static HTML on :8000      │
│  Database: SQLite (SQLAlchemy ORM)                               │
│  External: Cloudflare Tunnel → trumpgpt.arinjaff.com             │
└──────────────────────────────────────────────────────────────────┘
```

## Training Pipeline

### Markov Pipeline (every 6 hours, ~35 seconds)

```
Phase 0 → Social media refresh (Truth Social + Twitter/X, rebuild daily digests)
Phase 1 → Train order-3 Markov chain on all transcripts (~5s)
Phase 2 → Load tracked terms from DB (from Kalshi market titles)
Phase 3 → 2,000 Monte Carlo simulations × 5 scenarios
Phase 4 → Save predictions_latest.json
Phase 5 → Import to DB, blend with Pythia predictions if available
```

### Pi Fine-Tuning (nightly at 2 AM ET, ~75 min)

```
Phase 1 → Load corpus from DB (speeches + social media digests)
Phase 2 → Tokenize (512-token chunks)
Phase 3 → Load Pythia-160M + LoRA (rank 16, ~3.2M trainable params)
Phase 4 → Train 3 epochs (batch_size=1, grad_accum=8, gradient checkpointing)
Phase 5 → Save LoRA adapter + create ModelVersion record
Phase 6 → Run 200 Monte Carlo sims with fine-tuned model
Phase 7 → Save predictions_pythia.json → auto-blended on next Markov run
```

Model: **Pythia-160M** (EleutherAI, 160M params, ~640MB FP32)
LoRA: rank 16, targeting `query_key_value`, ~1.1GB peak RAM with gradient checkpointing
Training: CPU-only on Pi 4 (PyTorch 2.x supports ARM64 natively)

## Prediction Ensemble

7-signal weighted average, computed every 15 minutes:

| Signal | Weight | Description |
|--------|--------|-------------|
| Monte Carlo | 0.35 | Markov/LLM simulated speeches |
| **Social Velocity** | **0.20** | **Trending terms from Twitter/Truth Social (TF-IDF + freq delta)** |
| Frequency | 0.15 | Historical occurrence, 60-day half-life decay |
| News | 0.10 | Gemini 2.0 Flash Lite current events |
| Trend | 0.10 | Usage velocity + acceleration |
| Temporal | 0.05 | Day-of-week/seasonal (gated if dates poisoned) |
| Event | 0.05 | Event type ↔ term correlation |

Social media is now the primary terminology source — Twitter/Truth Social feeds are scraped
every 30 minutes, and the social_velocity signal (weight 0.20) captures surging terms
before they appear in speeches.

## Trading

- Fee-aware half-Kelly criterion (subtracts 4% Kalshi round-trip fee)
- Paper mode by default — flip to live via API/dashboard
- Position management: profit-take at +8c, stop-loss at -15c

## External Exposure

Only two external touchpoints:
1. **Website**: `trumpgpt.arinjaff.com` (via Cloudflare Tunnel) — full dashboard with all data, status, trading, health checks, and fine-tuning controls
2. **Daily Email**: 8 AM ET — P&L, trades, predictions, system health summary

## Known Issues

- **Terms table empty** — terms come from Kalshi market sync. No terms = no predictions. Need to run market sync first.
- **Bad dates in DB** — many speeches have `date = 2026-03-08` from failed date extraction during batch scrape.
- **HuggingFace login unresolved** — blocked Mac fine-tuning with Llama-3.2-1B. Resolved: Pi uses ungated Pythia-160M (no login needed).
