"""Social media term analysis for trending term detection.

Extracts trending terms from Twitter/Truth Social posts using TF-IDF and
frequency delta analysis. Feeds a 'social_velocity' signal into the
prediction ensemble, making social media the primary terminology source.
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from ..database.db import get_session
from ..database.models import Speech, Term


# Cache file for social trend scores
SOCIAL_TRENDS_PATH = os.path.join('data', 'predictions', 'social_trends.json')


class SocialMediaAnalyzer:
    """Analyze social media posts for trending terms and phrases.

    Computes two signals:
    1. TF-IDF: surfaces n-grams that are distinctive in recent posts
    2. Frequency delta: detects terms surging vs their baseline
    """

    def __init__(self):
        self._trend_scores = {}
        self._last_refresh = None
        self._load_cached_scores()

    def get_trend_score(self, term: str) -> Optional[float]:
        """Get social velocity score for a term (0.0-1.0).

        Returns None if no social media data available.
        """
        if not self._trend_scores:
            return None
        normalized = term.lower().strip()
        return self._trend_scores.get(normalized)

    def get_all_trends(self) -> dict:
        """Return all trend scores and metadata."""
        return {
            'scores': self._trend_scores.copy(),
            'last_refresh': self._last_refresh,
            'total_terms': len(self._trend_scores),
        }

    def refresh(self):
        """Recompute social media trend scores from recent posts.

        Called by the scheduler after each social media scrape.
        """
        try:
            recent_posts = self._get_recent_posts(days=7)
            baseline_posts = self._get_baseline_posts(days_start=8, days_end=60)

            if not recent_posts:
                logger.debug("Social media analyzer: no recent posts")
                return

            # Get tracked terms from DB
            tracked_terms = self._get_tracked_terms()

            # Compute frequency delta for tracked terms
            freq_scores = self._compute_frequency_delta(
                tracked_terms, recent_posts, baseline_posts
            )

            # Compute TF-IDF for emerging n-grams
            tfidf_scores = self._compute_tfidf_scores(
                tracked_terms, recent_posts, baseline_posts
            )

            # Merge scores (70% frequency delta, 30% TF-IDF relevance)
            merged = {}
            all_terms = set(list(freq_scores.keys()) + list(tfidf_scores.keys()))
            for term in all_terms:
                freq = freq_scores.get(term, 0.5)
                tfidf = tfidf_scores.get(term, 0.5)
                merged[term] = round(0.7 * freq + 0.3 * tfidf, 4)

            self._trend_scores = merged
            self._last_refresh = datetime.now().isoformat()
            self._save_cached_scores()

            logger.info(f"Social media analyzer: {len(merged)} term scores computed "
                        f"from {len(recent_posts)} recent posts")

        except Exception as e:
            logger.error(f"Social media analysis failed: {e}")

    def _get_recent_posts(self, days: int = 7) -> list[str]:
        """Get social media post texts from the last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with get_session() as session:
            posts = session.query(Speech.transcript).filter(
                Speech.speech_type == 'social_media',
                Speech.date >= cutoff,
                Speech.transcript.isnot(None),
            ).all()
            return [p[0] for p in posts if p[0]]

    def _get_baseline_posts(self, days_start: int = 8,
                            days_end: int = 60) -> list[str]:
        """Get baseline social media posts for comparison."""
        now = datetime.utcnow()
        start = now - timedelta(days=days_end)
        end = now - timedelta(days=days_start)
        with get_session() as session:
            posts = session.query(Speech.transcript).filter(
                Speech.speech_type == 'social_media',
                Speech.date >= start,
                Speech.date < end,
                Speech.transcript.isnot(None),
            ).all()
            return [p[0] for p in posts if p[0]]

    def _get_tracked_terms(self) -> list[str]:
        """Get all tracked terms from the database."""
        with get_session() as session:
            terms = session.query(Term).all()
            return [t.normalized_term.lower().strip() for t in terms]

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization for social media text."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) >= 2]

    def _extract_ngrams(self, tokens: list[str],
                        max_n: int = 3) -> list[str]:
        """Extract unigrams, bigrams, and trigrams."""
        ngrams = list(tokens)  # unigrams
        for n in range(2, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i + n]))
        return ngrams

    def _compute_frequency_delta(self, tracked_terms: list[str],
                                  recent_posts: list[str],
                                  baseline_posts: list[str]) -> dict:
        """Compute frequency surge for each tracked term.

        Compares term frequency in recent posts (7d) vs baseline (8-60d).
        Returns scores where >0.5 means surging, <0.5 means declining.
        """
        # Count term occurrences in recent and baseline
        recent_counts = Counter()
        baseline_counts = Counter()

        for text in recent_posts:
            text_lower = text.lower()
            for term in tracked_terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches:
                    recent_counts[term] += matches

        for text in baseline_posts:
            text_lower = text.lower()
            for term in tracked_terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches:
                    baseline_counts[term] += matches

        # Normalize by number of posts
        n_recent = max(1, len(recent_posts))
        n_baseline = max(1, len(baseline_posts))

        scores = {}
        for term in tracked_terms:
            recent_rate = recent_counts.get(term, 0) / n_recent
            baseline_rate = baseline_counts.get(term, 0) / n_baseline

            if baseline_rate > 0:
                ratio = recent_rate / baseline_rate
                # Sigmoid mapping: ratio of 3x maps to ~0.95
                score = 1 / (1 + math.exp(-1.5 * (ratio - 1)))
            elif recent_rate > 0:
                score = 0.85  # term appears recently but not in baseline
            else:
                score = 0.3  # never mentioned in social media

            scores[term] = round(score, 4)

        return scores

    def _compute_tfidf_scores(self, tracked_terms: list[str],
                               recent_posts: list[str],
                               baseline_posts: list[str]) -> dict:
        """Compute TF-IDF relevance scores for tracked terms.

        Terms that are distinctive in recent posts (high TF, low baseline DF)
        score higher.
        """
        if not recent_posts:
            return {}

        # Build document frequency from baseline
        baseline_df = Counter()
        n_baseline = max(1, len(baseline_posts))
        for text in baseline_posts:
            tokens = set(self._tokenize(text))
            ngrams = set(self._extract_ngrams(list(tokens), max_n=2))
            for ng in ngrams:
                baseline_df[ng] += 1

        # Compute TF in recent posts
        recent_tf = Counter()
        total_recent_tokens = 0
        for text in recent_posts:
            tokens = self._tokenize(text)
            total_recent_tokens += len(tokens)
            ngrams = self._extract_ngrams(tokens, max_n=2)
            recent_tf.update(ngrams)

        # Score tracked terms
        scores = {}
        for term in tracked_terms:
            tf = recent_tf.get(term, 0) / max(1, total_recent_tokens) * 1000
            df = baseline_df.get(term, 0)
            idf = math.log(n_baseline / (1 + df)) if n_baseline > 0 else 1.0

            tfidf = tf * idf
            # Normalize to 0-1 range (sigmoid)
            score = 1 / (1 + math.exp(-tfidf + 1))
            scores[term] = round(score, 4)

        return scores

    def _load_cached_scores(self):
        """Load cached trend scores from disk."""
        try:
            if os.path.exists(SOCIAL_TRENDS_PATH):
                with open(SOCIAL_TRENDS_PATH) as f:
                    data = json.load(f)
                self._trend_scores = data.get('scores', {})
                self._last_refresh = data.get('last_refresh')
        except Exception:
            pass

    def _save_cached_scores(self):
        """Persist trend scores to disk."""
        try:
            os.makedirs(os.path.dirname(SOCIAL_TRENDS_PATH), exist_ok=True)
            with open(SOCIAL_TRENDS_PATH, 'w') as f:
                json.dump({
                    'scores': self._trend_scores,
                    'last_refresh': self._last_refresh,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save social trends: {e}")


# Global singleton
social_media_analyzer = SocialMediaAnalyzer()
