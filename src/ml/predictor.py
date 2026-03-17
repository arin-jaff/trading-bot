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
    """

    def __init__(self):
        self.model_weights = {
            'frequency': 0.20,
            'temporal': 0.10,
            'trend': 0.15,
            'event_correlation': 0.15,
            'monte_carlo': 0.40,  # highest weight: Monte Carlo simulation
        }
        self._monte_carlo_predictions = None

    def predict_all_terms(self, event: Optional[dict] = None) -> list[dict]:
        """Generate predictions for all tracked terms."""
        # Load Monte Carlo predictions if available
        self._load_monte_carlo_predictions()

        predictions = []

        with get_session() as session:
            terms = session.query(Term).all()

            for term in terms:
                try:
                    pred = self._predict_term(session, term, event)
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"Prediction failed for term '{term.term}': {e}")

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

        # 2. Temporal patterns
        scores['temporal'] = self._temporal_score(session, term, event)

        # 3. Trend momentum
        scores['trend'] = self._trend_score(term)

        # 4. Event-type correlation
        scores['event_correlation'] = self._event_correlation_score(
            session, term, event
        )

        # 5. Monte Carlo prediction (local Markov chain or Colab LLM)
        mc_pred = self._monte_carlo_score(term)
        if mc_pred is not None:
            scores['monte_carlo'] = mc_pred
        else:
            # If no prediction, redistribute weight to other components
            scores['monte_carlo'] = None

        # Combined weighted score
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

        # Confidence based on data availability
        confidence = self._calculate_confidence(session, term)

        return {
            'term_id': term.id,
            'term': term.term,
            'probability': round(probability, 4),
            'confidence': round(confidence, 4),
            'component_scores': {k: round(v, 4) if v is not None else None for k, v in scores.items()},
            'model_name': 'ensemble_v1',
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
                        event: Optional[dict] = None) -> float:
        """Score based on temporal patterns (day of week, time of year)."""
        now = datetime.utcnow()
        current_dow = now.weekday()
        current_month = now.month

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
            dow_score = dow_counts[current_dow] / max(1, total) * 7  # normalize
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

    def _trend_score(self, term: Term) -> float:
        """Score based on recent trend direction."""
        trend = term.trend_score or 0.0

        # Map trend to [0, 1]: positive trend -> higher score
        # Sigmoid-like mapping
        import math
        return 1 / (1 + math.exp(-trend))

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
        """Calculate confidence level based on data quality and quantity."""
        occurrence_count = session.query(TermOccurrence).filter_by(
            term_id=term.id
        ).count()

        # More data points -> higher confidence
        data_confidence = min(1.0, occurrence_count / 50)

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
        """Save predictions to database."""
        with get_session() as session:
            for pred in predictions:
                term_pred = TermPrediction(
                    term_id=pred['term_id'],
                    event_id=event_id,
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
                for term in market.terms:
                    pred = latest_preds.get(term.id)
                    if not pred:
                        continue

                    market_yes_price = market.yes_price or 0.5
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
                            'kelly_fraction': self._kelly_criterion(
                                our_probability, market_yes_price
                            ),
                        })

        suggestions.sort(key=lambda x: abs(x['edge']), reverse=True)
        return suggestions

    def _kelly_criterion(self, prob: float, price: float) -> float:
        """Calculate Kelly criterion for optimal bet sizing.

        Returns the fraction of bankroll to bet.
        """
        if price <= 0 or price >= 1 or prob <= 0 or prob >= 1:
            return 0

        # For binary markets: f* = (p * (1-price) - (1-p) * price) / (1-price)
        # Simplified: f* = (p - price) / (1 - price) for YES bets
        if prob > price:  # YES bet
            kelly = (prob - price) / (1 - price)
        else:  # NO bet
            # Flip perspective
            no_prob = 1 - prob
            no_price = 1 - price
            kelly = (no_prob - no_price) / (1 - no_price)

        # Use fractional Kelly (half Kelly) for safety
        return max(0, min(0.25, kelly * 0.5))
