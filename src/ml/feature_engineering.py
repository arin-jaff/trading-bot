"""Feature engineering for the prediction model.

Extracts features from speech data, events, and current context
to feed into the prediction pipeline.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from loguru import logger
from sqlalchemy import func

from ..database.models import Term, TermOccurrence, Speech, TrumpEvent
from ..database.db import get_session


class FeatureEngineering:
    """Extracts and computes features for term prediction."""

    def build_feature_matrix(self, term_ids: list[int] = None) -> pd.DataFrame:
        """Build a feature matrix for all (or specified) terms.

        Returns DataFrame with one row per term, columns are features.
        """
        with get_session() as session:
            if term_ids:
                terms = session.query(Term).filter(Term.id.in_(term_ids)).all()
            else:
                terms = session.query(Term).all()

            rows = []
            for term in terms:
                features = self._compute_term_features(session, term)
                features['term_id'] = term.id
                features['term'] = term.term
                rows.append(features)

        return pd.DataFrame(rows)

    def _compute_term_features(self, session, term: Term) -> dict:
        """Compute all features for a single term."""
        features = {}

        # --- Frequency features ---
        total_speeches = session.query(Speech).filter_by(is_processed=True).count()
        speeches_with_term = session.query(TermOccurrence).filter_by(
            term_id=term.id
        ).count()
        total_mentions = session.query(func.sum(TermOccurrence.count)).filter_by(
            term_id=term.id
        ).scalar() or 0

        features['speech_frequency'] = speeches_with_term / max(1, total_speeches)
        features['total_mentions'] = total_mentions
        features['avg_mentions_per_speech'] = (
            total_mentions / max(1, speeches_with_term)
        )

        # --- Recency features ---
        occs_with_dates = session.query(TermOccurrence, Speech).join(Speech).filter(
            TermOccurrence.term_id == term.id
        ).order_by(Speech.date.desc()).all()

        if occs_with_dates:
            latest_date = occs_with_dates[0][1].date
            features['days_since_last_mention'] = (
                (datetime.utcnow() - latest_date).days if latest_date else 999
            )
        else:
            features['days_since_last_mention'] = 999

        # --- Trend features ---
        now = datetime.utcnow()
        periods = [7, 14, 30, 60, 90]
        for days in periods:
            cutoff = now - timedelta(days=days)
            count = session.query(func.sum(TermOccurrence.count)).join(Speech).filter(
                TermOccurrence.term_id == term.id,
                Speech.date >= cutoff
            ).scalar() or 0
            features[f'mentions_last_{days}d'] = count

        # Velocity: rate of change
        if features.get('mentions_last_30d', 0) > 0 and features.get('mentions_last_60d', 0) > 0:
            recent = features['mentions_last_30d']
            older = features['mentions_last_60d'] - recent
            features['velocity_30d'] = (recent - older) / max(1, older)
        else:
            features['velocity_30d'] = 0

        # Acceleration
        if features.get('mentions_last_14d', 0) > 0:
            v1 = features.get('mentions_last_7d', 0)
            v2 = features.get('mentions_last_14d', 0) - v1
            features['acceleration'] = (v1 - v2) / max(1, v2)
        else:
            features['acceleration'] = 0

        # --- Temporal features ---
        # Day-of-week distribution
        dow_dist = [0] * 7
        for occ, speech in occs_with_dates:
            if speech.date:
                dow_dist[speech.date.weekday()] += occ.count

        total_dow = sum(dow_dist)
        if total_dow > 0:
            dow_dist = [x / total_dow for x in dow_dist]
        features['dow_entropy'] = self._entropy(dow_dist)
        features['current_dow_weight'] = dow_dist[now.weekday()]

        # Speech type distribution
        type_counts = Counter()
        for occ, speech in occs_with_dates:
            if speech.speech_type:
                type_counts[speech.speech_type] += occ.count

        total_type = sum(type_counts.values())
        for stype in ['rally', 'press_conference', 'interview', 'remarks', 'address']:
            features[f'type_{stype}_pct'] = (
                type_counts.get(stype, 0) / max(1, total_type)
            )

        # --- Term characteristics ---
        features['term_length'] = len(term.term)
        features['term_word_count'] = len(term.term.split())
        features['is_compound'] = 1 if term.is_compound else 0

        # --- Burstiness ---
        # How bursty is this term? (high in some speeches, absent in others)
        if occs_with_dates:
            counts = [occ.count for occ, _ in occs_with_dates]
            features['mention_std'] = np.std(counts) if len(counts) > 1 else 0
            features['mention_max'] = max(counts)
            features['burstiness'] = features['mention_std'] / max(1, np.mean(counts))
        else:
            features['mention_std'] = 0
            features['mention_max'] = 0
            features['burstiness'] = 0

        return features

    def _entropy(self, probs: list[float]) -> float:
        """Calculate Shannon entropy."""
        probs = [p for p in probs if p > 0]
        if not probs:
            return 0
        return -sum(p * np.log2(p) for p in probs)

    def build_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Build training data from historical market outcomes.

        Uses settled markets as ground truth:
        - Feature matrix based on data available BEFORE the market resolved
        - Labels: 1 if term was said (market settled YES), 0 otherwise
        """
        from ..database.models import Market, market_term_association

        with get_session() as session:
            # Get settled markets
            settled = session.query(Market).filter(
                Market.result.in_(['yes', 'no'])
            ).all()

            if not settled:
                logger.warning("No settled markets for training data")
                return pd.DataFrame(), pd.Series()

            rows = []
            labels = []

            for market in settled:
                for term in market.terms:
                    features = self._compute_term_features(session, term)
                    features['term_id'] = term.id
                    rows.append(features)
                    labels.append(1 if market.result == 'yes' else 0)

        X = pd.DataFrame(rows)
        y = pd.Series(labels, name='outcome')

        # Drop non-numeric columns for training
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        return X[numeric_cols], y

    def get_context_features(self, event: dict = None) -> dict:
        """Get current context features for real-time prediction."""
        now = datetime.utcnow()

        features = {
            'hour_of_day': now.hour,
            'day_of_week': now.weekday(),
            'month': now.month,
            'is_weekend': 1 if now.weekday() >= 5 else 0,
        }

        if event:
            features['event_type'] = event.get('event_type', 'unknown')
            features['has_event'] = 1
        else:
            features['event_type'] = 'none'
            features['has_event'] = 0

        # Count upcoming events in next 24h
        with get_session() as session:
            upcoming_count = session.query(TrumpEvent).filter(
                TrumpEvent.start_time >= now,
                TrumpEvent.start_time <= now + timedelta(hours=24)
            ).count()
            features['events_next_24h'] = upcoming_count

        return features
