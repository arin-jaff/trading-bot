"""ML prediction engine for Trump term usage likelihood."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from sqlalchemy import func

from ..database.models import (
    Term, TermOccurrence, Speech, TrumpEvent, TermPrediction
)
from ..database.db import get_session


class TermPredictor:
    """Predicts likelihood of Trump using specific terms in upcoming events.

    Uses a combination of:
    1. Historical frequency analysis (statistical baseline)
    2. Temporal patterns (day-of-week, time-of-day, event-type effects)
    3. Topic-event correlation (certain events -> certain terms)
    4. Trend momentum (increasing/decreasing usage patterns)
    5. Monte Carlo predictions (Markov chain local or Colab LLM)
    6. News relevance (Gemini current events enrichment)
    """

    # Kalshi fee for edge filtering
    KALSHI_FEE_ROUND_TRIP = 0.04

    def __init__(self):
        self.model_weights = {
            'frequency': 0.20,
            'temporal': 0.05,       # reduced — often poisoned by bad dates
            'trend': 0.15,
            'event_correlation': 0.10,
            'monte_carlo': 0.40,
            'news_relevance': 0.10,  # 2A: Gemini current events signal
        }
        self._monte_carlo_predictions = None
        self._news_enricher = None
        self._correlation_matrix = None

    def predict_all_terms(self, event: Optional[dict] = None) -> list[dict]:
        """Generate predictions for all tracked terms."""
        # Load Monte Carlo predictions if available
        self._load_monte_carlo_predictions()

        # 2A: Load news enrichment
        self._load_news_enrichment()

        # 2D: Build correlation matrix
        self._build_correlation_matrix()

        predictions = []

        with get_session() as session:
            terms = session.query(Term).all()

            for term in terms:
                try:
                    pred = self._predict_term(session, term, event)
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"Prediction failed for term '{term.term}': {e}")

        # 2D: Apply correlation boost — if a high-confidence term is correlated,
        # boost the probability of its correlated partners
        predictions = self._apply_correlation_boost(predictions)

        # Sort by probability descending
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return predictions

    def _load_monte_carlo_predictions(self):
        """Load Monte Carlo predictions (from local Markov chain or Colab)."""
        try:
            from .colab_integration import ColabPredictor
            predictor = ColabPredictor()
            preds = predictor.get_predictions()
            if preds:
                self._monte_carlo_predictions = {
                    p['term'].lower().strip(): p for p in preds
                }
                logger.info(f"Loaded {len(preds)} Monte Carlo predictions")
            else:
                self._monte_carlo_predictions = {}
        except Exception as e:
            logger.debug(f"No Monte Carlo predictions available: {e}")
            self._monte_carlo_predictions = {}

    def _predict_term(self, session, term: Term,
                      event: Optional[dict] = None) -> dict:
        """Generate a combined prediction for a single term."""
        scores = {}

        # 1. Frequency-based probability
        scores['frequency'] = self._frequency_score(session, term)

        # 2. Temporal patterns (1B: gated on data quality)
        scores['temporal'] = self._temporal_score(session, term, event)

        # 3. Trend momentum
        scores['trend'] = self._trend_score(session, term)

        # 4. Event-type correlation
        scores['event_correlation'] = self._event_correlation_score(
            session, term, event
        )

        # 5. Monte Carlo prediction (local Markov chain or Colab LLM)
        mc_pred = self._monte_carlo_score(term)
        if mc_pred is not None:
            scores['monte_carlo'] = mc_pred
        else:
            scores['monte_carlo'] = None

        # 6. News relevance (2A: Gemini current events enrichment)
        scores['news_relevance'] = self._news_relevance_score(term)

        # Combined weighted score — None scores redistribute weight
        active_scores = {k: v for k, v in scores.items() if v is not None}
        active_weights = {k: self.model_weights.get(k, 0) for k in active_scores}
        total_weight = sum(active_weights.values())

        if total_weight > 0:
            probability = sum(
                active_scores[k] * active_weights[k] / total_weight
                for k in active_scores
            )
        else:
            probability = 0.5

        # Clamp to [0, 1]
        probability = max(0.0, min(1.0, probability))

        # Confidence based on data availability (1D: fixed threshold)
        confidence = self._calculate_confidence(session, term)

        return {
            'term_id': term.id,
            'term': term.term,
            'probability': round(probability, 4),
            'confidence': round(confidence, 4),
            'component_scores': {k: round(v, 4) if v is not None else None for k, v in scores.items()},
            'model_name': 'ensemble_v2',
        }

    def _frequency_score(self, session, term: Term) -> float:
        """Score based on how often the term appears across speeches.

        Uses recency-weighted frequency: recent mentions count more than
        older ones.  A term said 3 times last month scores higher than
        one said 50 times in 2017 but not since.
        """
        import math
        now = datetime.utcnow()

        # Get all occurrences with speech dates
        occs = session.query(TermOccurrence, Speech).join(Speech).filter(
            TermOccurrence.term_id == term.id,
            Speech.date.isnot(None),
        ).all()

        if not occs:
            return 0.1  # low score when never seen

        total_speeches = session.query(Speech).filter(
            Speech.is_processed == True
        ).count()

        if total_speeches == 0:
            return 0.5

        # Recency-weighted count: each occurrence is weighted by
        # exp(-days_ago / half_life).  Half-life of 60 days means
        # a mention 60 days ago counts half as much as one today.
        HALF_LIFE_DAYS = 60
        decay = math.log(2) / HALF_LIFE_DAYS

        weighted_count = 0.0
        raw_count = 0
        for occ, speech in occs:
            days_ago = max(0, (now - speech.date).days)
            weight = math.exp(-decay * days_ago)
            weighted_count += occ.count * weight
            raw_count += occ.count

        # Normalize: compare to what a term mentioned in every recent
        # speech would score
        recent_speeches = session.query(Speech).filter(
            Speech.is_processed == True,
            Speech.date >= now - timedelta(days=90),
        ).count()
        max_expected = max(1, recent_speeches)

        score = weighted_count / max_expected
        return min(1.0, score)

    def _temporal_score(self, session, term: Term,
                        event: Optional[dict] = None) -> Optional[float]:
        """Score based on temporal patterns (day of week, time of year).

        1B: Gate out poisoned data — if >30% of speeches have dates clustering
        around 2026-03-08 (the datetime.now() fallback from batch scraping),
        the temporal signal is unreliable and returns None to redistribute weight.
        """
        now = datetime.utcnow()
        current_dow = now.weekday()
        current_month = now.month

        # 1B: Check for date poisoning — detect any single date with suspiciously
        # many speeches (from batch scrapes that defaulted to datetime.now())
        total_speeches = session.query(Speech).filter(
            Speech.date.isnot(None)
        ).count()

        if total_speeches > 0:
            from sqlalchemy import func as sa_func
            # Find the most common date (by day)
            most_common = session.query(
                sa_func.date(Speech.date),
                sa_func.count(Speech.id),
            ).filter(
                Speech.date.isnot(None)
            ).group_by(
                sa_func.date(Speech.date)
            ).order_by(sa_func.count(Speech.id).desc()).first()

            if most_common and most_common[1] / total_speeches > 0.30:
                # A single date has >30% of all speeches — dates are poisoned
                return None

        # Get historical occurrences with speech dates
        occs = session.query(TermOccurrence, Speech).join(Speech).filter(
            TermOccurrence.term_id == term.id
        ).all()

        if not occs:
            return 0.5

        # Day-of-week analysis
        dow_counts = [0] * 7
        total = 0
        for occ, speech in occs:
            if speech.date:
                dow_counts[speech.date.weekday()] += occ.count
                total += occ.count

        if total > 0:
            dow_score = dow_counts[current_dow] / max(1, total) * 7
        else:
            dow_score = 0.5

        # Month analysis (seasonal patterns)
        month_counts = [0] * 12
        for occ, speech in occs:
            if speech.date:
                month_counts[speech.date.month - 1] += occ.count

        if total > 0:
            month_score = month_counts[current_month - 1] / max(1, total) * 12
        else:
            month_score = 0.5

        return min(1.0, (dow_score + month_score) / 2)

    def _trend_score(self, session, term: Term) -> float:
        """Score based on recent trend direction and weekly acceleration.

        Combines two signals:
        - Base trend: 30-day vs prior-30-day ratio (stored in term.trend_score)
        - Weekly acceleration: compares this week's mentions to the 4-week average
        """
        import math
        trend = term.trend_score or 0.0

        # Base: sigmoid mapping of 30-day trend
        base_score = 1 / (1 + math.exp(-trend))

        # Weekly acceleration: if we have occurrence data, check week-over-week
        try:
            now = datetime.utcnow()
            # This week (last 7 days)
            this_week = session.query(func.sum(TermOccurrence.count)).join(Speech).filter(
                TermOccurrence.term_id == term.id,
                Speech.date >= now - timedelta(days=7),
            ).scalar() or 0

            # Past 4 weeks average (days 7-35)
            past_4w = session.query(func.sum(TermOccurrence.count)).join(Speech).filter(
                TermOccurrence.term_id == term.id,
                Speech.date >= now - timedelta(days=35),
                Speech.date < now - timedelta(days=7),
            ).scalar() or 0

            weekly_avg = past_4w / 4.0 if past_4w > 0 else 0

            if weekly_avg > 0:
                # Acceleration: how much above/below the weekly average
                accel = (this_week - weekly_avg) / weekly_avg
                accel_score = 1 / (1 + math.exp(-accel * 2))  # steeper sigmoid
            else:
                accel_score = 0.6 if this_week > 0 else 0.4

            # Blend: 60% base trend, 40% weekly acceleration
            return base_score * 0.6 + accel_score * 0.4

        except Exception:
            return base_score

    def _event_correlation_score(self, session, term: Term,
                                  event: Optional[dict] = None) -> float:
        """Score based on correlation between term and event type."""
        if not event:
            return 0.5

        event_type = event.get('event_type', '').lower()
        if not event_type:
            return 0.5

        # Check historical: how often does this term appear in this event type?
        matching_speeches = session.query(TermOccurrence).join(Speech).filter(
            TermOccurrence.term_id == term.id,
            Speech.speech_type == event_type
        ).count()

        total_event_type = session.query(Speech).filter_by(
            speech_type=event_type, is_processed=True
        ).count()

        if total_event_type == 0:
            return 0.5

        return min(1.0, matching_speeches / max(1, total_event_type))

    def _monte_carlo_score(self, term: Term) -> Optional[float]:
        """Get probability from Monte Carlo simulation.

        This is the most powerful signal: simulated Trump speeches (via Markov
        chain or fine-tuned LLM) run many times to produce frequency-based
        probabilities.
        """
        if not self._monte_carlo_predictions:
            return None

        normalized = term.normalized_term.lower().strip()

        # Direct match
        if normalized in self._monte_carlo_predictions:
            return self._monte_carlo_predictions[normalized]['probability']

        # For compound terms, check sub-terms
        if term.is_compound and term.sub_terms:
            sub_probs = []
            for st in term.sub_terms:
                st_lower = st.lower().strip()
                if st_lower in self._monte_carlo_predictions:
                    sub_probs.append(self._monte_carlo_predictions[st_lower]['probability'])
            if sub_probs:
                # Any sub-term matching counts, so use max
                return max(sub_probs)

        return None

    def _calculate_confidence(self, session, term: Term) -> float:
        """Calculate confidence level based on data quality and quantity.

        1D fix: lowered threshold from 50 to 20 occurrences for max data confidence.
        Most terms have <20 occurrences, so the old threshold capped confidence
        at 0.3-0.4 and filtered out valid trades.
        """
        occurrence_count = session.query(TermOccurrence).filter_by(
            term_id=term.id
        ).count()

        # More data points -> higher confidence (1D: /20 instead of /50)
        data_confidence = min(1.0, occurrence_count / 20)

        # Recency: more recent data -> higher confidence
        latest = session.query(func.max(Speech.date)).join(TermOccurrence).filter(
            TermOccurrence.term_id == term.id
        ).scalar()

        if latest:
            days_ago = (datetime.utcnow() - latest).days
            recency_confidence = max(0, 1 - days_ago / 365)
        else:
            recency_confidence = 0

        return (data_confidence + recency_confidence) / 2

    def predict_with_llm(self, term: str, context: dict) -> dict:
        """Use LLM for nuanced prediction with contextual reasoning."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

            prompt = f"""You are analyzing the likelihood that Donald Trump will say the word/phrase "{term}" in an upcoming public appearance.

Context about the upcoming event:
- Event type: {context.get('event_type', 'unknown')}
- Event topic/title: {context.get('title', 'unknown')}
- Date: {context.get('date', 'unknown')}
- Location: {context.get('location', 'unknown')}

Historical data:
- This term has been used {context.get('total_occurrences', 0)} times across all analyzed speeches
- Recent trend: {'increasing' if context.get('trend_score', 0) > 0 else 'decreasing'} usage
- Appears in {context.get('frequency_pct', 0):.1f}% of speeches

Current political context to consider:
- Major current events and news cycles
- Trump's recent focus areas and talking points
- The specific audience and setting

Please provide:
1. A probability (0.0 to 1.0) that Trump will say this term
2. Your confidence level (0.0 to 1.0)
3. Brief reasoning (2-3 sentences)

Respond in JSON format:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "..."}}"""

            response = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=300,
                messages=[{'role': 'user', 'content': prompt}]
            )

            result_text = response.content[0].text
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")

        return {'probability': 0.5, 'confidence': 0.0, 'reasoning': 'LLM unavailable'}

    def save_predictions(self, predictions: list[dict],
                         event_id: Optional[int] = None,
                         target_date: Optional[datetime] = None):
        """Save predictions to database, tagged with the active model version."""
        with get_session() as session:
            # Get active model version ID for tagging
            from ..database.models import ModelVersion
            active_mv = session.query(ModelVersion).filter_by(
                is_active=True
            ).order_by(ModelVersion.created_at.desc()).first()
            mv_id = active_mv.id if active_mv else None

            for pred in predictions:
                term_pred = TermPrediction(
                    term_id=pred['term_id'],
                    event_id=event_id,
                    model_version_id=mv_id,
                    model_name=pred.get('model_name', 'ensemble_v1'),
                    probability=pred['probability'],
                    confidence=pred.get('confidence', 0),
                    reasoning=pred.get('reasoning', ''),
                    features_used=pred.get('component_scores'),
                    target_date=target_date or datetime.utcnow(),
                )
                session.add(term_pred)

        logger.info(f"Saved {len(predictions)} predictions")

    def get_trading_suggestions(self, min_edge: float = 0.05) -> list[dict]:
        """Compare predictions to market prices and suggest trades.

        A trade is suggested when our predicted probability differs
        significantly from the market-implied probability.
        """
        from ..database.models import Market, market_term_association

        suggestions = []

        with get_session() as session:
            # Get latest predictions
            latest_preds = {}
            terms = session.query(Term).all()
            for term in terms:
                latest = session.query(TermPrediction).filter_by(
                    term_id=term.id
                ).order_by(TermPrediction.created_at.desc()).first()
                if latest:
                    latest_preds[term.id] = latest

            # Get active markets
            markets = session.query(Market).filter(
                Market.status.in_(['active', 'open'])
            ).all()

            for market in markets:
                market_yes_price = market.yes_price or 0.5

                # Skip markets at 1c/99c — already decided, untradeable
                if market_yes_price >= 0.99 or market_yes_price <= 0.01:
                    continue

                for term in market.terms:
                    pred = latest_preds.get(term.id)
                    if not pred:
                        continue

                    our_probability = pred.probability

                    edge = our_probability - market_yes_price

                    if abs(edge) >= min_edge:
                        suggestions.append({
                            'market_ticker': market.kalshi_ticker,
                            'market_title': market.title,
                            'term': term.term,
                            'market_yes_price': market_yes_price,
                            'our_probability': our_probability,
                            'edge': round(edge, 4),
                            'confidence': pred.confidence,
                            'suggested_side': 'yes' if edge > 0 else 'no',
                            'suggested_action': 'buy',
                            'reasoning': pred.reasoning,
                            'volume': market.volume or 0,
                            'close_time': market.close_time.isoformat() if market.close_time else None,
                            'kelly_fraction': self._kelly_criterion(
                                our_probability, market_yes_price
                            ),
                        })

        suggestions.sort(key=lambda x: abs(x['edge']), reverse=True)
        return suggestions

    def _kelly_criterion(self, prob: float, price: float) -> float:
        """Calculate fee-aware Kelly criterion for optimal bet sizing (1A).

        Subtracts Kalshi round-trip fee from edge before computing Kelly fraction.
        Returns the fraction of bankroll to bet.
        """
        if price <= 0 or price >= 1 or prob <= 0 or prob >= 1:
            return 0

        if prob > price:  # YES bet
            raw_edge = prob - price
            # 1A: Subtract fee from edge — if net edge is non-positive, don't bet
            net_edge = raw_edge - self.KALSHI_FEE_ROUND_TRIP
            if net_edge <= 0:
                return 0
            kelly = net_edge / (1 - price)
        else:  # NO bet
            no_prob = 1 - prob
            no_price = 1 - price
            raw_edge = no_prob - no_price
            net_edge = raw_edge - self.KALSHI_FEE_ROUND_TRIP
            if net_edge <= 0:
                return 0
            kelly = net_edge / (1 - no_price)

        # Use fractional Kelly (half Kelly) for safety
        return max(0, min(0.25, kelly * 0.5))

    # ─── 2A: News Relevance ─────────────────────────────────────────────

    def _load_news_enrichment(self):
        """Load the news enricher singleton."""
        if self._news_enricher is not None:
            return
        try:
            from .news_enrichment import news_enricher
            self._news_enricher = news_enricher
        except Exception as e:
            logger.debug(f"News enrichment not available: {e}")
            self._news_enricher = None

    def _news_relevance_score(self, term: Term) -> Optional[float]:
        """Get news relevance score for a term from Gemini enrichment (2A)."""
        if not self._news_enricher:
            return None

        try:
            boost = self._news_enricher.get_term_boost(term.normalized_term)
            if boost is not None:
                return boost
            # Also check sub-terms for compound terms
            if term.is_compound and term.sub_terms:
                boosts = []
                for st in term.sub_terms:
                    b = self._news_enricher.get_term_boost(st.lower().strip())
                    if b is not None:
                        boosts.append(b)
                if boosts:
                    return max(boosts)
        except Exception:
            pass
        return None

    # ─── 2D: Term Correlation Matrix ────────────────────────────────────

    def _build_correlation_matrix(self):
        """Build co-occurrence matrix from TermOccurrence table (2D)."""
        if self._correlation_matrix is not None:
            return

        try:
            with get_session() as session:
                # Get all term occurrences grouped by speech
                from collections import defaultdict
                speech_terms = defaultdict(set)

                occs = session.query(
                    TermOccurrence.term_id, TermOccurrence.speech_id
                ).all()

                for term_id, speech_id in occs:
                    speech_terms[speech_id].add(term_id)

                # Count co-occurrences
                co_occur = defaultdict(lambda: defaultdict(int))
                term_counts = defaultdict(int)

                for speech_id, term_ids in speech_terms.items():
                    for tid in term_ids:
                        term_counts[tid] += 1
                        for other_tid in term_ids:
                            if tid != other_tid:
                                co_occur[tid][other_tid] += 1

                # Normalize to correlation scores (Jaccard similarity)
                self._correlation_matrix = {}
                for tid, others in co_occur.items():
                    self._correlation_matrix[tid] = {}
                    for other_tid, count in others.items():
                        union = term_counts[tid] + term_counts[other_tid] - count
                        if union > 0:
                            self._correlation_matrix[tid][other_tid] = count / union

        except Exception as e:
            logger.debug(f"Could not build correlation matrix: {e}")
            self._correlation_matrix = {}

    def _apply_correlation_boost(self, predictions: list[dict]) -> list[dict]:
        """Boost predictions based on correlated high-confidence terms (2D)."""
        if not self._correlation_matrix:
            return predictions

        # Find high-confidence predictions (prob > 0.7, confidence > 0.5)
        high_conf = {
            p['term_id']: p for p in predictions
            if p['probability'] > 0.7 and p['confidence'] > 0.5
        }

        if not high_conf:
            return predictions

        for pred in predictions:
            tid = pred['term_id']
            if tid in high_conf:
                continue  # already high confidence

            corr = self._correlation_matrix.get(tid, {})
            boosts = []
            for hc_tid, hc_pred in high_conf.items():
                corr_score = corr.get(hc_tid, 0)
                if corr_score > 0.3:  # meaningfully correlated
                    # Boost proportional to correlation and the anchor's probability
                    boost = corr_score * (hc_pred['probability'] - 0.5) * 0.2
                    boosts.append(boost)

            if boosts:
                total_boost = sum(boosts) / len(boosts)
                pred['probability'] = max(0.0, min(1.0,
                    round(pred['probability'] + total_boost, 4)
                ))

        return predictions

    # ─── 1C: Prediction Performance Tracker ─────────────────────────────

    def evaluate_accuracy(self) -> dict:
        """Evaluate prediction accuracy against settled market outcomes (1C).

        Returns Brier score, hit rate, and calibration data.
        """
        with get_session() as session:
            from ..database.models import Market, market_term_association

            # Get settled markets with results
            settled = session.query(Market).filter(
                Market.result.in_(['yes', 'no'])
            ).all()

            if not settled:
                return {'error': 'No settled markets found', 'settled_count': 0}

            data_points = []

            for market in settled:
                actual = 1.0 if market.result == 'yes' else 0.0

                for term in market.terms:
                    # Find the most recent prediction before close time
                    pred_query = session.query(TermPrediction).filter(
                        TermPrediction.term_id == term.id,
                    )
                    if market.close_time:
                        pred_query = pred_query.filter(
                            TermPrediction.created_at <= market.close_time
                        )
                    pred = pred_query.order_by(
                        TermPrediction.created_at.desc()
                    ).first()

                    if pred:
                        data_points.append({
                            'predicted': pred.probability,
                            'actual': actual,
                            'confidence': pred.confidence,
                            'term': term.term,
                            'market': market.kalshi_ticker,
                            'model_version_id': pred.model_version_id,
                        })

                        # Update was_correct field
                        pred.was_correct = (
                            (pred.probability > 0.5 and actual == 1.0) or
                            (pred.probability <= 0.5 and actual == 0.0)
                        )

            if not data_points:
                return {'error': 'No prediction-settlement pairs found',
                        'settled_count': len(settled)}

            # Compute metrics
            brier_scores = [(d['predicted'] - d['actual']) ** 2 for d in data_points]
            brier_score = sum(brier_scores) / len(brier_scores)

            correct = sum(
                1 for d in data_points
                if (d['predicted'] > 0.5 and d['actual'] == 1.0) or
                   (d['predicted'] <= 0.5 and d['actual'] == 0.0)
            )
            hit_rate = correct / len(data_points)

            # Calibration by 10% buckets
            buckets = {}
            for i in range(10):
                low, high = i * 0.1, (i + 1) * 0.1
                bucket_points = [d for d in data_points if low <= d['predicted'] < high]
                if bucket_points:
                    avg_pred = sum(d['predicted'] for d in bucket_points) / len(bucket_points)
                    avg_actual = sum(d['actual'] for d in bucket_points) / len(bucket_points)
                    buckets[f'{int(low*100)}-{int(high*100)}%'] = {
                        'avg_predicted': round(avg_pred, 4),
                        'avg_actual': round(avg_actual, 4),
                        'count': len(bucket_points),
                    }

            # Per-version accuracy breakdown
            from ..database.models import ModelVersion
            version_metrics = []
            # Group data_points by model_version_id
            by_version = {}
            for d in data_points:
                vid = d.get('model_version_id')
                if vid not in by_version:
                    by_version[vid] = []
                by_version[vid].append(d)

            for vid, vpoints in by_version.items():
                v_brier = sum((d['predicted'] - d['actual']) ** 2 for d in vpoints) / len(vpoints)
                v_correct = sum(
                    1 for d in vpoints
                    if (d['predicted'] > 0.5 and d['actual'] == 1.0) or
                       (d['predicted'] <= 0.5 and d['actual'] == 0.0)
                )
                v_hit = v_correct / len(vpoints)

                version_str = None
                if vid:
                    mv = session.query(ModelVersion).filter_by(id=vid).first()
                    version_str = mv.version if mv else None

                version_metrics.append({
                    'version': version_str or 'untagged',
                    'model_version_id': vid,
                    'brier_score': round(v_brier, 4),
                    'hit_rate': round(v_hit, 4),
                    'data_points': len(vpoints),
                })

            version_metrics.sort(key=lambda x: x['version'])

            return {
                'brier_score': round(brier_score, 4),
                'hit_rate': round(hit_rate, 4),
                'total_data_points': len(data_points),
                'settled_markets': len(settled),
                'calibration': buckets,
                'ready_for_live': brier_score < 0.25,
                'per_version': version_metrics,
            }
