# TrumpGPT Optimization — Feature Changelog

Implemented 2026-03-16. Organized in 4 tiers by ROI.

---

## Tier 1: Critical Fixes

### 1A. Fee-Aware Kelly Criterion

**Problem:** Kelly formula ignored Kalshi's ~2% per-side fee (4% round-trip). A 5% edge was treated as 5% when it's really 1% net.

**Changes:**
- `src/ml/predictor.py` — `_kelly_criterion()` subtracts round-trip fee from edge before computing Kelly fraction. Returns 0 if net edge is non-positive.
- `src/kalshi/trading_bot.py` — `generate_suggestions()` rejects positions where `|edge| - 0.04 <= 0`. Arbitrage bounds updated from `0.98/1.02` to fee-aware thresholds (`0.92/1.08`).
- `TradingBot` class constants: `KALSHI_FEE_PER_SIDE = 0.02`, `KALSHI_FEE_ROUND_TRIP = 0.04`.

**Verification:** Run `predictor.get_trading_suggestions()` — no suggestions with `|edge| < 0.04`.

### 1B. Gate Out Poisoned Temporal Signal

**Problem:** Most speeches have wrong dates (`2026-03-08` from `datetime.now()` fallback during batch scraping). Temporal score computed day-of-week/month histograms on fake dates.

**Changes:**
- `src/ml/predictor.py` — `_temporal_score()` counts speeches within ±1 day of `2026-03-08`. If >30% of all speeches fall in that window, returns `None` instead of a score, redistributing the temporal weight (5%) to other signals.

**Verification:** `predictor.predict_all_terms()` — temporal scores are `None` when dates are poisoned.

### 1C. Prediction Performance Tracker

**Problem:** No feedback loop. No way to know if the model is good or bad.

**Changes:**
- `src/ml/predictor.py` — New method `evaluate_accuracy()`: joins `TermPrediction` with settled `Market` records, computes hit rate, Brier score, and calibration curve by 10% bucket.
- `src/api/server.py` — New endpoint `GET /api/model/accuracy` returns accuracy metrics.
- `static/index.html` — System tab shows accuracy panel: Brier score, hit rate, data point count, live-ready gate (Brier < 0.25), and calibration table.
- `scripts/backfill_settlements.py` — Standalone script to backfill `was_correct` on historical predictions and optionally fit Platt scaling calibration.

**Verification:** `GET /api/model/accuracy` — returns Brier score and calibration data. `python scripts/backfill_settlements.py --dry-run` for preview.

### 1D. Confidence Scaling Fix

**Problem:** Confidence required 50 occurrences for max score. Most terms have <20, capping confidence at 0.3-0.4 and filtering them out via `min_confidence=0.3`.

**Changes:**
- `src/ml/predictor.py` — `_calculate_confidence()`: threshold lowered from `occurrence_count / 50` to `occurrence_count / 20`.
- `src/kalshi/trading_bot.py` — `_calculate_position_size()`: Kelly fraction scaled by confidence (`adjusted_kelly *= confidence`). Low-confidence predictions get smaller positions rather than being filtered entirely.

---

## Tier 2: Better Predictions

### 2A. Current Events Enrichment via Gemini

**Problem:** The model had no idea what's in the news cycle. Trump's vocabulary is driven by current events.

**Changes:**
- `src/ml/news_enrichment.py` — **NEW FILE**. `NewsEnricher` class calls Gemini 2.0 Flash Lite API (uses `GEMINI_API_KEY` env var). Asks for top 15 likely Trump talking points with relevance scores. Results cached for 1 hour. Module-level singleton `news_enricher`.
- `src/ml/predictor.py` — Added as 6th ensemble signal: `news_relevance` (weight `0.10`). Temporal weight reduced from `0.10` to `0.05` to compensate (temporal is often poisoned anyway). `_news_relevance_score()` checks exact and substring matches against Gemini's talking points.
- `src/scheduler.py` — New job `_refresh_news_enrichment()` runs every 1 hour.
- `src/config.py` — Added `gemini_api_key` field.
- `static/index.html` — Pipeline tab scheduled jobs table shows "News Enrichment" row.

**Cost:** ~$0.0024/day (24 calls × $0.0001/call).

### 2C. Per-Event Scenario Weighting

**Problem:** Monte Carlo used hardcoded scenario weights (40% rally, 25% press conf) even when the next event type is known.

**Changes:**
- `src/ml/local_pipeline.py` — New method `_get_event_scenario_weights()`: queries the next confirmed `TrumpEvent`, maps `event_type` to scenario weight dicts (e.g., rally → 85% rally sims). Returns `None` to use defaults if no event found.
- `run_full_pipeline()` now calls `_get_event_scenario_weights()` and passes result to `trainer.run_monte_carlo(scenario_weights=...)`.

**Event type mappings:**
| Event Type | Rally Weight | Press Conf | Chopper | Fox | Social |
|---|---|---|---|---|---|
| rally | 85% | 5% | 3% | 5% | 2% |
| press_conference | 5% | 80% | 5% | 5% | 5% |
| interview/fox | 5% | 5% | 5% | 80% | 5% |
| state_dinner | 10% | 60% | 10% | 10% | 10% |

### 2D. Term Correlation Matrix

**Problem:** "tariff" and "China" are highly correlated but treated independently. Bot could double-bet on correlated terms.

**Changes:**
- `src/ml/predictor.py` — `_build_correlation_matrix()`: computes Jaccard co-occurrence similarity from `TermOccurrence` table (which terms appear in the same speeches).
- `_apply_correlation_boost()`: after initial predictions, terms correlated (Jaccard > 0.3) with high-confidence predictions (prob > 0.7, confidence > 0.5) get a proportional probability boost.

---

## Tier 3: Smarter Trading

### 3A. New Market Front-Running

**Problem:** When Kalshi creates a new market, it opens at ~50c before price discovery. Our model may already have a prediction.

**Changes:**
- `src/kalshi/market_sync.py` — `sync_markets()` now tracks `new_market_tickers` in return stats.
- `src/scheduler.py` — `_sync_markets()` detects new tickers, creates alerts via `alert_manager`, and sends email notification listing new markets.

### 3B. Profit-Taking & Stop-Loss

**Problem:** Held all positions to expiry with no active management.

**Changes:**
- `src/kalshi/trading_bot.py` — New method `manage_positions()`:
  - **Take profit:** If unrealized gain > 2× round-trip fee (>8c/contract), sell 50% to lock profit.
  - **Stop loss:** If position down >15c from entry, sell all to limit losses.
  - New helper `_execute_sell()` for placing sell orders.
- `src/scheduler.py` — New job `_manage_positions()` runs every 5 minutes.
- `static/index.html` — Pipeline tab shows "Position Mgmt" in scheduled jobs.

### 3C. Time-to-Close Decay

**Problem:** Markets closing in 30 minutes vs 5 days should be sized differently.

**Changes:**
- `src/kalshi/trading_bot.py` — `_calculate_position_size()` applies time-based multiplier:
  - Markets closing in <2 hours: `0.7×` Kelly (less time to recover if wrong)
  - Markets closing in >5 days (120h): `0.5×` Kelly (more uncertainty)
  - Uses `close_time` field now included in trading suggestions.

### 3D. Liquidity Filter

**Problem:** Low-volume markets have wide spreads — edge gets eaten by market impact.

**Changes:**
- `src/kalshi/trading_bot.py` — New config parameter `min_volume = 50`. `generate_suggestions()` filters out markets below this threshold. `_calculate_position_size()` caps position to 10% of market's daily volume.
- `src/ml/predictor.py` — `get_trading_suggestions()` now includes `volume` and `close_time` in suggestion dicts.

---

## Tier 4: Data Quality

### 4B. Kalshi Settlement Data as Ground Truth

**Problem:** No labeled dataset for calibration.

**Changes:**
- `scripts/backfill_settlements.py` — **NEW FILE**. Standalone script:
  1. Queries all settled markets (`result` in `yes`/`no`)
  2. Matches each to the latest `TermPrediction` before `close_time`
  3. Updates `was_correct` field on prediction records
  4. Prints summary: accuracy, Brier score, calibration by 10% bucket
  5. Optional Platt scaling via `scipy.optimize.curve_fit` (fits sigmoid `a*p + b`)

**Usage:**
```bash
python scripts/backfill_settlements.py            # backfill + write to DB
python scripts/backfill_settlements.py --dry-run   # preview without writing
```

### 4C. Duplicate Speech Detection

**Problem:** Same speech scraped from multiple sources inflates term frequencies.

**Changes:**
- `src/scraper/speech_scraper.py` — `_save_speech()` now checks for duplicate transcripts before saving. Normalizes first 40 characters of the transcript to lowercase, searches existing speeches for a match. Skips saving if a duplicate from a different source is found.

---

## Updated Ensemble Weights

| Signal | Old Weight | New Weight | Notes |
|---|---|---|---|
| `frequency` | 0.20 | 0.20 | Unchanged |
| `temporal` | 0.10 | 0.05 | Reduced — often poisoned by bad dates, gated out entirely when >30% poisoned |
| `trend` | 0.15 | 0.15 | Unchanged |
| `event_correlation` | 0.15 | 0.10 | Slightly reduced |
| `monte_carlo` | 0.40 | 0.40 | Unchanged — dominant signal |
| `news_relevance` | — | 0.10 | **NEW** — Gemini current events |

When a signal returns `None` (e.g., temporal is gated out, news enrichment unavailable), its weight is redistributed proportionally to the remaining active signals.

## Updated Scheduler Jobs

| Job | Interval | New? |
|---|---|---|
| Market Sync | 5 min | Updated — detects new markets (3A) |
| Speech Scrape | 2 hours | Updated — duplicate detection (4C) |
| Event Tracking | 30 min | — |
| Live Event Check | 1 min | — |
| Predictions | 15 min | Updated — 6 signals, correlation boost |
| Trading Check | 5 min | Updated — fee-aware, liquidity filter |
| Local Pipeline | 6 hours | Updated — per-event scenario weights (2C) |
| Arbitrage Scan | 10 min | Updated — fee-aware bounds (1A) |
| **News Enrichment** | **1 hour** | **NEW** (2A) |
| **Position Management** | **5 min** | **NEW** (3B) |
| Daily Digest | 8 AM UTC | — |

## New API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/model/accuracy` | GET | Prediction accuracy: Brier score, hit rate, calibration (1C) |

## New Files

| File | Purpose |
|---|---|
| `src/ml/news_enrichment.py` | Gemini 2.0 Flash Lite current events enrichment (2A) |
| `scripts/backfill_settlements.py` | One-time settlement backfill + Platt calibration (4B) |

## Modified Files

| File | Changes |
|---|---|
| `src/kalshi/trading_bot.py` | Fee-aware Kelly, position management, time decay, liquidity filter |
| `src/ml/predictor.py` | Temporal gate, confidence fix, news signal, correlation matrix, accuracy tracker, fee-aware Kelly |
| `src/ml/local_pipeline.py` | Per-event scenario weighting |
| `src/kalshi/market_sync.py` | New market detection |
| `src/scheduler.py` | News enrichment job, position management job, new market alerts |
| `src/api/server.py` | Accuracy endpoint |
| `src/config.py` | `gemini_api_key` field |
| `src/scraper/speech_scraper.py` | Duplicate transcript detection |
| `static/index.html` | Accuracy display, updated scheduled jobs table |
