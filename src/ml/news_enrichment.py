"""Current events enrichment via Gemini 2.0 Flash.

Queries Gemini for Trump's likely talking points based on recent news,
returning relevance scores that can boost prediction probabilities.
Uses structured JSON outputs (response_schema) for guaranteed parsing
and injects live RSS headlines for context beyond the training cutoff.
Results are persisted to disk and refreshed every 5 days.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


CACHE_FILE = Path("data/news_cache.json")

# Free RSS feeds for live headline context
RSS_FEEDS = [
    "https://feeds.apnews.com/rss/politics",
    "https://moxie.foxnews.com/google-publisher/politics.xml",
    "https://feeds.reuters.com/reuters/politicsNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
]


class NewsEnricher:
    """Fetches Trump's likely talking points from current news via Gemini API.

    Uses structured JSON outputs for reliable parsing and injects live
    RSS headlines to overcome the model's training data cutoff.
    Results are cached to disk for 5 days to stay well within free tier limits.
    """

    CACHE_TTL_SECONDS = 5 * 24 * 3600  # 5 days

    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._cache_reasoning: Dict[str, str] = {}
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
                self._cache_reasoning = data.get("reasoning", {})
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
                "reasoning": self._cache_reasoning,
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

    def _fetch_headlines(self) -> List[str]:
        """Fetch current political headlines from free RSS feeds."""
        headlines = []
        try:
            import feedparser
            import requests
        except ImportError:
            logger.debug("feedparser/requests not available for RSS headlines")
            return []

        for feed_url in RSS_FEEDS:
            try:
                resp = requests.get(feed_url, timeout=8,
                                    headers={'User-Agent': 'TrumpBot/1.0'})
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[:5]:
                    title = entry.get('title', '').strip()
                    if title:
                        headlines.append(title)
            except Exception as e:
                logger.debug(f"RSS fetch failed for {feed_url}: {e}")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for h in headlines:
            key = h.lower()
            if key not in seen:
                seen.add(key)
                unique.append(h)

        logger.debug(f"Fetched {len(unique)} unique headlines from {len(RSS_FEEDS)} RSS feeds")
        return unique[:20]

    def refresh(self) -> Dict[str, float]:
        """Call Gemini API to get current talking points. Skips if cache is fresh.

        Uses structured JSON output (response_schema) for guaranteed parsing,
        and injects live RSS headlines for context beyond the training cutoff.
        """
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

            # Fetch live headlines for context
            headlines = self._fetch_headlines()
            headlines_block = ""
            if headlines:
                numbered = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(headlines))
                headlines_block = (
                    f"\n\nCurrent political headlines (last 24 hours):\n{numbered}\n\n"
                    "Use these headlines to ground your predictions in current events.\n"
                )

            prompt = (
                "You are analyzing what Donald Trump is most likely to say in his next "
                "speech or public appearance. Consider ongoing political events, controversies, "
                "policy topics, people in the news, and his known rhetorical patterns."
                f"{headlines_block}\n"
                "Return the top 25 words or short phrases he is most likely to mention, "
                "with a relevance score (0.0-1.0) and brief reasoning for each."
            )

            # Structured output schema — guarantees valid JSON, no regex needed
            response_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "relevance_score": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["term", "relevance_score", "reasoning"],
                },
            }

            model = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

            response = model.generate_content(prompt)
            items = json.loads(response.text)

            result: Dict[str, float] = {}
            reasoning: Dict[str, str] = {}
            for item in items:
                term = str(item.get('term', '')).lower().strip()
                relevance = max(0.0, min(1.0, float(item.get('relevance_score', 0))))
                reason = str(item.get('reasoning', ''))
                if term:
                    result[term] = relevance
                    reasoning[term] = reason

            self._cache = result
            self._cache_reasoning = reasoning
            self._cache_timestamp = time.time()
            self._save_disk_cache()
            headline_note = f" (with {len(headlines)} live headlines)" if headlines else ""
            logger.info(f"News enrichment refreshed via Gemini: {len(result)} talking points{headline_note} (next refresh in 5 days)")
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
