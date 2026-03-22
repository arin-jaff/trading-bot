"""Current events enrichment via Gemini 2.0 Flash Lite.

Queries Gemini for Trump's likely talking points based on recent news,
returning relevance scores that can boost prediction probabilities.
"""

import os
import json
import re
import time
from typing import Dict, Optional

from loguru import logger


class NewsEnricher:
    """Fetches Trump's likely talking points from current news via Gemini API.

    Results are cached for 1 hour to avoid redundant API calls.
    """

    CACHE_TTL_SECONDS = 3600  # 1 hour
    BACKOFF_SECONDS = 3600  # 1 hour backoff after quota/rate limit failure

    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._cache_timestamp: float = 0.0
        self._api_key: str = os.getenv('GEMINI_API_KEY', '')
        self._backoff_until: float = 0.0  # skip calls until this timestamp

    def _is_cache_valid(self) -> bool:
        """Check if cached results are still fresh."""
        if not self._cache:
            return False
        return (time.time() - self._cache_timestamp) < self.CACHE_TTL_SECONDS

    def refresh(self) -> Dict[str, float]:
        """Call Gemini API to get current talking points.

        Returns:
            Dict mapping lowercase term -> relevance score (0.0-1.0).
            Empty dict on failure.
        """
        if not self._api_key:
            logger.warning("GEMINI_API_KEY not set - news enrichment disabled")
            return {}

        if time.time() < self._backoff_until:
            remaining = int(self._backoff_until - time.time())
            logger.debug(f"News enrichment in backoff, skipping ({remaining}s remaining)")
            return self._cache

        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            prompt = (
                "Based on the news from the last 48 hours, what are the top 15 words or "
                "short phrases that Donald Trump is most likely to say in his next speech or "
                "public appearance? Consider ongoing political events, controversies, policy "
                "topics, and people in the news.\n\n"
                "Return ONLY a JSON array of objects with 'term' (the word/phrase) and "
                "'relevance' (a float from 0.0 to 1.0 indicating how likely he is to mention it). "
                "Example: [{\"term\": \"border\", \"relevance\": 0.95}]\n\n"
                "JSON array:"
            )

            response = model.generate_content(prompt)
            text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if not json_match:
                logger.warning(f"Gemini response did not contain valid JSON array: {text[:200]}")
                return {}

            items = json.loads(json_match.group())

            result: Dict[str, float] = {}
            for item in items:
                if isinstance(item, dict) and 'term' in item and 'relevance' in item:
                    term = str(item['term']).lower().strip()
                    relevance = float(item['relevance'])
                    relevance = max(0.0, min(1.0, relevance))
                    if term:
                        result[term] = relevance

            self._cache = result
            self._cache_timestamp = time.time()
            logger.info(f"News enrichment refreshed: {len(result)} talking points loaded")
            return result

        except ImportError:
            logger.warning("google-generativeai package not installed - news enrichment disabled")
            return {}
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                self._backoff_until = time.time() + self.BACKOFF_SECONDS
                logger.warning(f"Gemini quota/rate limit hit - backing off for 1 hour. Skipping news enrichment.")
            else:
                logger.warning(f"Gemini API call failed: {e}")
            return self._cache

    def get_talking_points(self) -> Dict[str, float]:
        """Get cached talking points, refreshing if stale.

        Returns:
            Dict mapping lowercase term -> relevance score (0.0-1.0).
        """
        if not self._is_cache_valid():
            self.refresh()
        return self._cache

    def get_term_boost(self, term: str) -> Optional[float]:
        """Get the relevance score for a specific term.

        Args:
            term: The term to look up (case-insensitive).

        Returns:
            Relevance score (0.0-1.0) if found, None otherwise.
        """
        points = self.get_talking_points()
        normalized = term.lower().strip()

        # Exact match
        if normalized in points:
            return points[normalized]

        # Substring match — if the term appears within any talking point or vice versa
        for cached_term, score in points.items():
            if normalized in cached_term or cached_term in normalized:
                return score

        return None


# Module-level singleton
news_enricher = NewsEnricher()
