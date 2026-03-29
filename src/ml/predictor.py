"""ML prediction engine for Trump term usage likelihood."""

import os
import json
import numpy as np
import pandas as pd
import gc
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
            'frequency': 0.15,
            'temporal': 0.05,
            'trend': 0.10,
            'event_correlation': 0.05,
            'monte_carlo': 0.35,
            'news_relevance': 0.10,
            'social_velocity': 0.20,
        }
        self._monte_carlo_predictions = None
        self._news_enricher = None
        self._social_analyzer = None
        self._correlation_matrix = None

    def predict_all_terms(self, event: Optional[dict] = None) -> list[dict]:
        """Generate predictions for all tracked terms."""
        self._load_monte_carlo_predictions()
        self._load_news_enrichment()
        self._load_social_analyzer()
        self._build_correlation_matrix()

        predictions = []

        with get_session() as session:
            terms = session.query(Term).all()

            prefetched = self._prefetch_data(session, terms, event)

            for term in terms:
                try:
                    pred = self._predict_term(session, term, event, prefetched)
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"Prediction failed for term '{term.term}': {e}")

        predictions = self._apply_correlation_boost(predictions)
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Free heavy local vars and trigger GC explicitly to prevent API starving
        del prefetched
        gc.collect()
        
        return predictions

    def _load_monte_carlo_predictions(self):
        import math
        import glob as _glob

        self._monte_carlo_predictions = {}
        pred_dir = os.path.join('data', 'predictions')
        latest_path = os.path.join(pred_dir, 'predictions_latest.json')

        if not os.path.exists(latest_path):
            files = sorted(_glob.glob(os.path.join(pred_dir, 'predictions_*.json')), reverse=True)
            if files:
                latest_path = files[0]
            else:
                return

        try:
            with open(latest_path) as f:
                data = json.load(f)

            predictions = data.get('term_predictions', [])
            if not predictions:
                return

            with get_session() as session:
                term_map = {t.normalized_term: t.id for t in session.query(Term).all()}
                for pred in predictions:
                    pred['term_id'] = term_map.get(pred['term'].lower().strip())

            sim_params = data.get('simulation_params', {})
            snippet_words = sim_params.get('avg_words_per_speech', 430)
            target_words = 7000
            if snippet_words < target_words * 0.8:
                for pred in predictions:
                    avg_mentions = pred.get('avg_mentions_per_speech')
                    if avg_mentions and avg_mentions > 0:
                        lam = avg_mentions / snippet_words
                        pred['raw_probability'] = pred['probability']
                        pred['probability'] = round(1.0 - math.exp(-lam * target_words), 4)

            self._monte_carlo_predictions = {
                p['term'].lower().strip(): p for p in predictions
            }
            logger.info(f"Loaded {len(predictions)} Monte Carlo predictions from {latest_path}")

        except Exception as e:
            logger.debug(f"No Monte Carlo predictions available: {e}")
            self._monte_carlo_predictions = {}

    def _prefetch_data(self, session, terms, event: Optional[dict] = None) -> dict:
        import math
        from collections import defaultdict
        now = datetime.utcnow()

        data = {}

        data['total_speeches'] = session.query(Speech).filter(
            Speech.is_processed == True
        ).count()

        data['recent_speeches_90d'] = session.query(Speech).filter(
            Speech.is_processed == True,
            Speech.date >= now - timedelta(days=90),
        ).count()

        total_with_dates = session.query(Speech).filter(
            Speech.date.isnot(None)
        ).count()
        data['dates_poisoned'] = False
        if total_with_dates > 0:
            from sqlalchemy import func as sa_func
            most_common = session.query(
                sa_func.date(Speech.date),
                sa_func.count(Speech.id),
            ).filter(
                Speech.date.isnot(None)
            ).group_by(
                sa_func.date(Speech.date)
            ).order_by(sa_func.count(Speech.id).desc()).first()
            if most_common and most_common[1] / total_with_dates > 0.30:
                data['dates_poisoned'] = True

        all_occs = session.query(
            TermOccurrence.term_id,
            TermOccurrence.count,
            TermOccurrence.speech_id,
            Speech.date,
            Speech.speech_type,
        ).join(Speech).filter(
            Speech.date.isnot(None),
        ).all()

        occs_by_term = defaultdict(list)
        for term_id, count, speech_id, date, speech_type in all_occs:
            occs_by_term[term_id].append({
                'count': count,
                'speech_id': speech_id,
                'date': date,
                'speech_type': speech_type,
            })
        data['occs_by_term'] = occs_by_term

        this_week_cutoff = now - timedelta(days=7)
        past_4w_start = now - timedelta(days=35)

        this_week_rows = session.query(
            TermOccurrence.term_id,
            func.sum(TermOccurrence.count),
        ).join(Speech).filter(
            Speech.date >= this_week_cutoff,
        ).group_by(TermOccurrence.term_id).all()
        data['trend_this_week'] = {tid: total for tid, total in this_week_rows}

        past_4w_rows = session.query(
            TermOccurrence.term_id,
            func.sum(TermOccurrence.count),
        ).join(Speech).filter(
            Speech.date >= past_4w_start,
            Speech.date < this_week_cutoff,
        ).group_by(TermOccurrence.term_id).all()
        data['trend_past_4w'] = {tid: total for tid, total in past_4w_rows}

        if event and event.get('event_type'):
            event_type = event['event_type'].lower()
            data['event_type_total'] = session.query(Speech).filter_by(
                speech_type=event_type, is_processed=True
            ).count()

            event_occs = session.query(
                TermOccurrence.term_id,
                func.count(TermOccurrence.id),
            ).join(Speech).filter(
                Speech.speech_type == event_type,
            ).group_by(TermOccurrence.term_id).all()
            data['event_occs_by_term'] = {tid: cnt for tid, cnt in event_occs}
        else:
            data['event_type_total'] = 0
            data['event_occs_by_term'] = {}

        occ_counts = session.query(
            TermOccurrence.term_id,
            func.count(TermOccurrence.id),
        ).group_by(TermOccurrence.term_id).all()
        data['occ_counts'] = {tid: cnt for tid, cnt in occ_counts}

        latest_dates = session.query(
            TermOccurrence.term_id,
            func.max(Speech.date),
        ).join(Speech).group_by(TermOccurrence.term_id).all()
        data['latest_date_by_term'] = {tid: dt for tid, dt in latest_dates}

        logger.debug(f"Prefetched prediction data: {len(occs_by_term)} terms with occurrences")
        return data

    def _predict_term(self, session, term: Term,
                      event: Optional[dict] = None,
                      prefetched: Optional[dict] = None) -> dict:
        scores = {}

        scores['frequency'] = self._frequency_score(session, term, prefetched)
        scores['temporal'] = self._temporal_score(session, term, event, prefetched)
        scores['trend'] = self._trend_score(session, term, prefetched)
        scores['event_correlation'] = self._event_correlation_score(
            session, term, event, prefetched
        )

        mc_pred = self._monte_carlo_score(term)
        if mc_pred is not None:
            scores['monte_carlo'] = mc_pred
        else:
            scores['monte_carlo'] = None

        scores['news_relevance'] = self._news_relevance_score(term)
        scores['social_velocity'] = self._social_velocity_score(term)

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

        probability = max(0.0, min(1.0, probability))
        confidence = self._calculate_confidence(session, term, prefetched)

        return {
            'term_id': term.id,
            'term': term.term,
            'probability': round(probability, 4),
            'confidence': round(confidence, 4),
            'component_scores': {k: round(v, 4) if v is not None else None for k, v in scores.items()},
            'model_name': 'ensemble_v2',
        }

    def _frequency_score(self, session, term: Term,
                         prefetched: Optional[dict] = None) -> float:
        import math
        now = datetime.utcnow()

        occs = prefetched['occs_by_term'].get(term.id, []) if prefetched else None
        if occs is None:
            rows = session.query(TermOccurrence, Speech).join(Speech).filter(
                TermOccurrence.term_id == term.id,
                Speech.date.isnot(None),
            ).all()
            occs = [{'count': o.count, 'date': s.date} for o, s in rows]

        if not occs:
            return 0.1

        total_speeches = prefetched['total_speeches'] if prefetched else (
            session.query(Speech).filter(Speech.is_processed == True).count()
        )
        if total_speeches == 0:
            return 0.5

        HALF_LIFE_DAYS = 60
        decay = math.log(2) / HALF_LIFE_DAYS

        weighted_count = 0.0
        for occ in occs:
            days_ago = max(0, (now - occ['date']).days)
            weight = math.exp(-decay * days_ago)
            weighted_count += occ['count'] * weight

        recent_speeches = prefetched['recent_speeches_90d'] if prefetched else (
            session.query(Speech).filter(
                Speech.is_processed == True,
                Speech.date >= now - timedelta(days=90),
            ).count()
        )
        max_expected = max(1, recent_speeches)

        score = weighted_count / max_expected
        return min(1.0, score)

    def _temporal_score(self, session, term: Term,
                        event: Optional[dict] = None,
                        prefetched: Optional[dict] = None) -> Optional[float]:
        now = datetime.utcnow()
        current_dow = now.weekday()
        current_month = now.month

        if prefetched:
            if prefetched['dates_poisoned']:
                return None
        else:
            total_speeches = session.query(Speech).filter(
                Speech.date.isnot(None)
            ).count()
            if total_speeches > 0:
                from sqlalchemy import func as sa_func
                most_common = session.query(
                    sa_func.date(Speech.date),
                    sa_func.count(Speech.id),
                ).filter(
                    Speech.date.isnot(None)
                ).group_by(
                    sa_func.date(Speech.date)
                ).order_by(sa_func.count(Speech.id).desc()).first()
                if most_common and most_common[1] / total_speeches > 0.30:
                    return None

        occs = prefetched['occs_by_term'].get(term.id, []) if prefetched else None
        if occs is None:
            rows = session.query(TermOccurrence, Speech).join(Speech).filter(
                TermOccurrence.term_id == term.id
            ).all()
            occs = [{'count': o.count, 'date': s.date} for o, s in rows]

        if not occs:
            return 0.5

        dow_counts = [0] * 7
        total = 0
        for occ in occs:
            if occ['date']:
                dow_counts[occ['date'].weekday()] += occ['count']
                total += occ['count']

        if total > 0:
            dow_score = dow_counts[current_dow] / max(1, total) * 7
        else:
            dow_score = 0.5

        month_counts = [0] * 12
        for occ in occs:
            if occ['date']:
                month_counts[occ['date'].month - 1] += occ['count']

        if total > 0:
            month_score = month_counts[current_month - 1] / max(1, total) * 12
        else:
            month_score = 0.5

        return min(1.0, (dow_score + month_score) / 2)

    def _trend_score(self, session, term: Term,
                     prefetched: Optional[dict] = None) -> float:
        import math
        trend = term.trend_score or 0.0

        base_score = 1 / (1 + math.exp(-trend))

        try:
            if prefetched:
                this_week = prefetched['trend_this_week'].get(term.id, 0)
                past_4w = prefetched['trend_past_4w'].get(term.id, 0)
            else:
                now = datetime.utcnow()
                this_week = session.query(func.sum(TermOccurrence.count)).join(Speech).filter(
                    TermOccurrence.term_id == term.id,
                    Speech.date >= now - timedelta(days=7),
                ).scalar() or 0
                past_4w = session.query(func.sum(TermOccurrence.count)).join(Speech).filter(
                    TermOccurrence.term_id == term.id,
                    Speech.date >= now - timedelta(days=35),
                    Speech.date < now - timedelta(days=7),
                ).scalar() or 0

            weekly_avg = past_4w / 4.0 if past_4w > 0 else 0

            if weekly_avg > 0:
                accel = (this_week - weekly_avg) / weekly_avg
                accel_score = 1 / (1 + math.exp(-accel * 2))
            else:
                accel_score = 0.6 if this_week > 0 else 0.4

            return base_score * 0.6 + accel_score * 0.4

        except Exception:
            return base_score

    def _event_correlation_score(self, session, term: Term,
                                  event: Optional[dict] = None,
                                  prefetched: Optional[dict] = None) -> float:
        if not event:
            return 0.5

        event_type = event.get('event_type', '').lower()
        if not event_type:
            return 0.5

        if prefetched:
            total_event_type = prefetched['event_type_total']
            matching_speeches = prefetched['event_occs_by_term'].get(term.id, 0)
        else:
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
        if not self._monte_carlo_predictions:
            return None

        normalized = term.normalized_term.lower().strip()

        if normalized in self._monte_carlo_predictions:
            return self._monte_carlo_predictions[normalized]['probability']

        if term.is_compound and term.sub_terms:
            sub_probs = []
            for st in term.sub_terms:
                st_lower = st.lower().strip()
                if st_lower in self._monte_carlo_predictions:
                    sub_probs.append(self._monte_carlo_predictions[st_lower]['probability'])
            if sub_probs:
                return max(sub_probs)

        return None

    def _calculate_confidence(self, session, term: Term,
                              prefetched: Optional[dict] = None) -> float:
        if prefetched:
            occurrence_count = prefetched['occ_counts'].get(term.id, 0)
            latest = prefetched['latest_date_by_term'].get(term.id)
        else:
            occurrence_count = session.query(TermOccurrence).filter_by(
                term_id=term.id
            ).count()
            latest = session.query(func.max(Speech.date)).join(TermOccurrence).filter(
                TermOccurrence.term_id == term.id
            ).scalar()

        data_confidence = min(1.0, occurrence_count / 20)

        if latest:
            days_ago = (datetime.utcnow() - latest).days
            recency_confidence = max(0, 1 - days_ago / 365)
        else:
            recency_confidence = 0

        return (data_confidence + recency_confidence) / 2

    def predict_with_llm(self, term: str, context: dict) -> dict:
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
        with get_session() as session:
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
        from ..database.models import Market, market_term_association

        suggestions = []

        with get_session() as session:
            latest_preds = {}
            terms = session.query(Term).all()
            for term in terms:
                latest = session.query(TermPrediction).filter_by(
                    term_id=term.id
                ).order_by(TermPrediction.created_at.desc()).first()
                if latest:
                    latest_preds[term.id] = latest

            markets = session.query(Market).filter(
                Market.status.in_(['active', 'open'])
            ).all()

            for market in markets:
                market_yes_price = market.yes_price or 0.5

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
        if price <= 0 or price >= 1 or prob <= 0 or prob >= 1:
            return 0

        if prob > price:
            raw_edge = prob - price
            net_edge = raw_edge - self.KALSHI_FEE_ROUND_TRIP
            if net_edge <= 0:
                return 0
            kelly = net_edge / (1 - price)
        else:
            no_prob = 1 - prob
            no_price = 1 - price
            raw_edge = no_prob - no_price
            net_edge = raw_edge - self.KALSHI_FEE_ROUND_TRIP
            if net_edge <= 0:
                return 0
            kelly = net_edge / (1 - no_price)

        return max(0, min(0.25, kelly * 0.5))

    def _load_news_enrichment(self):
        if self._news_enricher is not None:
            return
        try:
            from .news_enrichment import news_enricher
            self._news_enricher = news_enricher
        except Exception as e:
            logger.debug(f"News enrichment not available: {e}")
            self._news_enricher = None

    def _news_relevance_score(self, term: Term) -> Optional[float]:
        if not self._news_enricher:
            return None

        try:
            boost = self._news_enricher.get_term_boost(term.normalized_term)
            if boost is not None:
                return boost
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

    def _load_social_analyzer(self):
        if self._social_analyzer is not None:
            return
        try:
            from .social_media_analyzer import social_media_analyzer
            self._social_analyzer = social_media_analyzer
        except Exception as e:
            logger.debug(f"Social media analyzer not available: {e}")
            self._social_analyzer = None

    def _social_velocity_score(self, term: Term) -> Optional[float]:
        if not self._social_analyzer:
            return None

        try:
            score = self._social_analyzer.get_trend_score(term.normalized_term)
            if score is not None:
                return score
            if term.is_compound and term.sub_terms:
                sub_scores = []
                for st in term.sub_terms:
                    s = self._social_analyzer.get_trend_score(st.lower().strip())
                    if s is not None:
                        sub_scores.append(s)
                if sub_scores:
                    return max(sub_scores)
        except Exception:
            pass
        return None

    def _build_correlation_matrix(self):
        if self._correlation_matrix is not None:
            return

        try:
            with get_session() as session:
                from collections import defaultdict
                speech_terms = defaultdict(set)

                occs = session.query(
                    TermOccurrence.term_id, TermOccurrence.speech_id
                ).all()

                for term_id, speech_id in occs:
                    speech_terms[speech_id].add(term_id)

                co_occur = defaultdict(lambda: defaultdict(int))
                term_counts = defaultdict(int)

                for speech_id, term_ids in speech_terms.items():
                    for tid in term_ids:
                        term_counts[tid] += 1
                        for other_tid in term_ids:
                            if tid != other_tid:
                                co_occur[tid][other_tid] += 1

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
        if not self._correlation_matrix:
            return predictions

        high_conf = {
            p['term_id']: p for p in predictions
            if p['probability'] > 0.7 and p['confidence'] > 0.5
        }

        if not high_conf:
            return predictions

        for pred in predictions:
            tid = pred['term_id']
            if tid in high_conf:
                continue

            corr = self._correlation_matrix.get(tid, {})
            boosts = []
            for hc_tid, hc_pred in high_conf.items():
                corr_score = corr.get(hc_tid, 0)
                if corr_score > 0.3:
                    boost = corr_score * (hc_pred['probability'] - 0.5) * 0.2
                    boosts.append(boost)

            if boosts:
                total_boost = sum(boosts) / len(boosts)
                pred['probability'] = max(0.0, min(1.0,
                    round(pred['probability'] + total_boost, 4)
                ))

        return predictions

    def evaluate_accuracy(self) -> dict:
        with get_session() as session:
            from ..database.models import Market, market_term_association

            settled = session.query(Market).filter(
                Market.result.in_(['yes', 'no'])
            ).all()

            if not settled:
                return {'error': 'No settled markets found', 'settled_count': 0}

            data_points = []

            for market in settled:
                actual = 1.0 if market.result == 'yes' else 0.0

                for term in market.terms:
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

                        pred.was_correct = (
                            (pred.probability > 0.5 and actual == 1.0) or
                            (pred.probability <= 0.5 and actual == 0.0)
                        )

            if not data_points:
                return {'error': 'No prediction-settlement pairs found',
                        'settled_count': len(settled)}

            brier_scores = [(d['predicted'] - d['actual']) ** 2 for d in data_points]
            brier_score = sum(brier_scores) / len(brier_scores)

            correct = sum(
                1 for d in data_points
                if (d['predicted'] > 0.5 and d['actual'] == 1.0) or
                   (d['predicted'] <= 0.5 and d['actual'] == 0.0)
            )
            hit_rate = correct / len(data_points)

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

            from ..database.models import ModelVersion
            version_metrics = []
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