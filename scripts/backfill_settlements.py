#!/usr/bin/env python3
"""Tier 4B: Backfill settlement data as ground truth for prediction calibration.

Queries all settled Kalshi markets, matches them to the predictions that were
active at settlement time, marks was_correct on TermPrediction records, and
computes accuracy / calibration statistics.

Optionally fits a Platt scaling (sigmoid) calibration curve if scipy is available.

Usage:
    python scripts/backfill_settlements.py
    python scripts/backfill_settlements.py --dry-run   # preview without writing
"""

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from sqlalchemy import and_

from src.database.db import get_session, init_db
from src.database.models import Market, Term, TermPrediction, market_term_association


def find_latest_prediction_before(session, term_id: int, before_dt) -> TermPrediction | None:
    """Find the most recent prediction for a term created before a given datetime."""
    return (
        session.query(TermPrediction)
        .filter(
            TermPrediction.term_id == term_id,
            TermPrediction.created_at <= before_dt,
        )
        .order_by(TermPrediction.created_at.desc())
        .first()
    )


def backfill_settlements(dry_run: bool = False):
    """Main backfill logic: match settled markets to predictions and score them."""
    init_db()

    calibration_data = []  # list of (predicted_probability, actual_outcome)
    total_settled = 0
    total_matched = 0
    total_correct = 0
    total_updated = 0
    skipped_no_terms = 0
    skipped_no_prediction = 0

    with get_session() as session:
        # Get all settled markets with a definitive result
        settled_markets = (
            session.query(Market)
            .filter(Market.result.in_(['yes', 'no']))
            .all()
        )

        total_settled = len(settled_markets)

        if total_settled == 0:
            logger.warning("No settled markets found in the database.")
            return

        logger.info(f"Found {total_settled} settled markets to process.")

        for market in settled_markets:
            actual_outcome = 1 if market.result == 'yes' else 0

            if not market.terms:
                skipped_no_terms += 1
                logger.debug(
                    f"Market {market.kalshi_ticker} has no associated terms, skipping."
                )
                continue

            if not market.close_time:
                logger.debug(
                    f"Market {market.kalshi_ticker} has no close_time, skipping."
                )
                continue

            for term in market.terms:
                prediction = find_latest_prediction_before(
                    session, term.id, market.close_time
                )

                if prediction is None:
                    skipped_no_prediction += 1
                    logger.debug(
                        f"No prediction found for term '{term.term}' "
                        f"before {market.close_time} (market {market.kalshi_ticker})"
                    )
                    continue

                total_matched += 1

                # Determine correctness:
                # Prediction says YES if probability > 0.5, NO if <= 0.5
                predicted_yes = prediction.probability > 0.5
                actual_yes = market.result == 'yes'
                was_correct = predicted_yes == actual_yes

                if was_correct:
                    total_correct += 1

                calibration_data.append((prediction.probability, actual_outcome))

                # Update the prediction record
                if prediction.was_correct != was_correct:
                    total_updated += 1
                    if not dry_run:
                        prediction.was_correct = was_correct

                logger.debug(
                    f"  Term '{term.term}' | "
                    f"predicted={prediction.probability:.3f} | "
                    f"result={market.result} | "
                    f"correct={was_correct}"
                )

        if not dry_run:
            session.commit()
            logger.info(f"Committed {total_updated} prediction updates to database.")
        else:
            logger.info(f"DRY RUN: would have updated {total_updated} predictions.")

    # --- Summary Statistics ---
    print("\n" + "=" * 60)
    print("SETTLEMENT BACKFILL SUMMARY")
    print("=" * 60)
    print(f"Total settled markets:          {total_settled}")
    print(f"Skipped (no associated terms):  {skipped_no_terms}")
    print(f"Skipped (no prediction found):  {skipped_no_prediction}")
    print(f"Predictions matched:            {total_matched}")
    print(f"Predictions updated:            {total_updated}")

    if total_matched == 0:
        print("\nNo predictions matched to settled markets. Nothing to evaluate.")
        return

    accuracy = total_correct / total_matched
    print(f"\nOverall accuracy:               {accuracy:.1%} ({total_correct}/{total_matched})")

    # Brier score
    brier_score = sum(
        (pred - actual) ** 2 for pred, actual in calibration_data
    ) / len(calibration_data)
    print(f"Brier score:                    {brier_score:.4f}  (lower is better, 0.25 = coin flip)")

    # Calibration by bucket
    print("\n--- Calibration by Probability Bucket ---")
    print(f"{'Bucket':<12} {'Count':>6} {'Avg Predicted':>14} {'Actual Rate':>12} {'Gap':>8}")
    print("-" * 54)

    buckets = defaultdict(list)
    for pred, actual in calibration_data:
        # Bucket: 0-10%, 10-20%, ..., 90-100%
        bucket_idx = min(int(pred * 10), 9)
        bucket_label = f"{bucket_idx * 10}-{(bucket_idx + 1) * 10}%"
        buckets[bucket_label].append((pred, actual))

    for i in range(10):
        label = f"{i * 10}-{(i + 1) * 10}%"
        entries = buckets.get(label, [])
        if not entries:
            print(f"{label:<12} {'--':>6} {'--':>14} {'--':>12} {'--':>8}")
            continue
        avg_pred = sum(p for p, _ in entries) / len(entries)
        actual_rate = sum(a for _, a in entries) / len(entries)
        gap = actual_rate - avg_pred
        print(f"{label:<12} {len(entries):>6} {avg_pred:>13.1%} {actual_rate:>11.1%} {gap:>+7.1%}")

    # --- Optional: Platt Scaling ---
    try:
        _fit_platt_scaling(calibration_data)
    except ImportError:
        logger.info("scipy not available -- skipping Platt scaling calibration.")
    except Exception as e:
        logger.warning(f"Platt scaling failed: {e}")


def _fit_platt_scaling(calibration_data):
    """Fit a sigmoid (Platt scaling) to map raw predictions to calibrated probabilities."""
    import numpy as np
    from scipy.optimize import curve_fit

    predictions = np.array([p for p, _ in calibration_data])
    actuals = np.array([a for _, a in calibration_data])

    if len(predictions) < 10:
        logger.info("Too few data points for Platt scaling (need at least 10).")
        return

    def sigmoid(x, a, b):
        return 1.0 / (1.0 + np.exp(-(a * x + b)))

    try:
        popt, pcov = curve_fit(sigmoid, predictions, actuals, p0=[1.0, 0.0], maxfev=5000)
        a, b = popt

        print("\n--- Platt Scaling Calibration ---")
        print(f"Fitted sigmoid: P_calibrated = 1 / (1 + exp(-({a:.4f} * p + {b:.4f})))")

        # Show calibrated Brier score
        calibrated = sigmoid(predictions, a, b)
        calibrated_brier = np.mean((calibrated - actuals) ** 2)
        raw_brier = np.mean((predictions - actuals) ** 2)
        print(f"Raw Brier score:        {raw_brier:.4f}")
        print(f"Calibrated Brier score: {calibrated_brier:.4f}")
        improvement = (raw_brier - calibrated_brier) / raw_brier * 100
        print(f"Improvement:            {improvement:+.1f}%")

        # Show some example mappings
        print("\nExample calibration mappings:")
        for raw_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            cal_p = sigmoid(np.array([raw_p]), a, b)[0]
            print(f"  {raw_p:.0%} -> {cal_p:.1%}")

    except RuntimeError as e:
        logger.warning(f"Sigmoid curve fitting did not converge: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill settlement outcomes onto TermPrediction records."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to the database.",
    )
    args = parser.parse_args()

    logger.info(f"Starting settlement backfill (dry_run={args.dry_run})")
    backfill_settlements(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
