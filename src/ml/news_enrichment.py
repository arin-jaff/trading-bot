"""Current events enrichment via Gemini 2.0 Flash.

Queries Gemini for Trump's likely talking points based on recent news,
returning relevance scores that can boost prediction probabilities.
Results are persisted to disk and refreshed every 5 days.
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


CACHE_FILE = Path("data/news_cache.json")


class NewsEnricher:
    """Fetches Trump's likely talking points from current news via Gemini API.

    Results are cached to disk for 5 days to stay well within free tier limits.
    """

    CACHE_TTL_SECONDS = 5 * 24 * 3600  # 5 days

    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._cache_timestamp: float = 0.0
        self._quota_backoff_until: float = 0.0  # don't retry until this time
        self._api_key: str = os.getenv('GEMINI_API_KEY', '')
        self._load_disk_cache()

    def _load_disk_cache(self):
        """Load cached results from disk on startup."""
        try:
            if CACHE_FILE.exists():
                data = json.loads(CACHE_FILE.read_text())
                self._cache = data.get("talking_points", {})
                self._cache_timestamp = data.get("timestamp", 0.0)
                age_days = (time.time() - self._cache_timestamp) / 86400
                logger.info(f"News cache loaded from disk: {len(self._cache)} terms, {age_days:.1f} days old")
        except Exception as e:
            logger.warning(f"Failed to load news cache from disk: {e}")

    def _save_disk_cache(self):
        """Persist cache to disk."""
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps({
                "talking_points": self._cache,
                "timestamp": self._cache_timestamp,
                "refreshed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(self._cache_timestamp)),
            }, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save news cache to disk: {e}")

    def _is_cache_valid(self) -> bool:
        # If in quota backoff, treat cache as valid to stop retrying
        if time.time() < self._quota_backoff_until:
            return True
        if not self._cache:
            return False
        return (time.time() - self._cache_timestamp) < self.CACHE_TTL_SECONDS

    def refresh(self) -> Dict[str, float]:
        """Call Gemini API to get current talking points. Skips if cache is fresh."""
        if self._is_cache_valid():
            age_days = (time.time() - self._cache_timestamp) / 86400
            logger.debug(f"News cache still valid ({age_days:.1f} days old), skipping Gemini call")
            return self._cache

        if not self._api_key:
            logger.warning("GEMINI_API_KEY not set - news enrichment disabled")
            return {}

        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            prompt = (
                "Based on the news from the last week, what are the top 25 words or "
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

            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if not json_match:
                logger.warning(f"Gemini response did not contain valid JSON array: {text[:200]}")
                return self._cache

            items = json.loads(json_match.group())

            result: Dict[str, float] = {}
            for item in items:
                if isinstance(item, dict) and 'term' in item and 'relevance' in item:
                    term = str(item['term']).lower().strip()
                    relevance = max(0.0, min(1.0, float(item['relevance'])))
                    if term:
                        result[term] = relevance

            self._cache = result
            self._cache_timestamp = time.time()
            self._save_disk_cache()
            logger.info(f"News enrichment refreshed via Gemini: {len(result)} talking points (next refresh in 5 days)")
            return result

        except ImportError:
            logger.warning("google-generativeai package not installed - news enrichment disabled")
            return {}
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower():
                self._quota_backoff_until = time.time() + 3600  # back off 1 hour
                logger.warning("Gemini quota hit - backing off 1 hour. Using existing cache.")
            else:
                logger.warning(f"Gemini API call failed: {e}")
            return self._cache

    def get_talking_points(self) -> Dict[str, float]:
        """Get cached talking points, refreshing only if older than 5 days."""
        if not self._is_cache_valid():
            self.refresh()
        return self._cache

    def get_term_boost(self, term: str) -> Optional[float]:
        """Get the relevance score for a specific term."""
        points = self.get_talking_points()
        normalized = term.lower().strip()

        if normalized in points:
            return points[normalized]

        for cached_term, score in points.items():
            if normalized in cached_term or cached_term in normalized:
                return score

        return None


# Module-level singleton
news_enricher = NewsEnricher()
