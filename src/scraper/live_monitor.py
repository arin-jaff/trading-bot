"""Live speech monitoring - detects when Trump is speaking and tracks terms in real-time."""

import os
import re
import time
import threading
from datetime import datetime
from typing import Optional, Callable
from loguru import logger

import requests
from bs4 import BeautifulSoup

from ..database.models import Term, TrumpEvent
from ..database.db import get_session


class LiveSpeechMonitor:
    """Monitors live Trump speeches and tracks term mentions in real-time.

    Uses multiple detection methods:
    1. YouTube live stream detection
    2. White House live feed
    3. C-SPAN live detection
    4. RSS/News alerts for breaking speech events
    """

    def __init__(self):
        self.is_monitoring = False
        self._thread = None
        self._callbacks: list[Callable] = []
        self._detected_terms: dict[str, int] = {}
        self._current_event: Optional[dict] = None
        self._compiled_patterns: list[tuple] = []  # [(term_name, compiled_regex), ...]
        self._patterns_built_at: float = 0.0
        self._last_yt_search: float = 0.0  # rate-limit YouTube API calls
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        })

    def register_callback(self, callback: Callable):
        """Register a callback for live term detection events.

        Callback receives: (term: str, count: int, source: str, timestamp: datetime)
        """
        self._callbacks.append(callback)

    def start_monitoring(self):
        """Start live monitoring in a background thread."""
        if self.is_monitoring:
            logger.warning("Monitor already running")
            return

        self.is_monitoring = True
        self._detected_terms = {}
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Live speech monitor started")

    def stop_monitoring(self):
        """Stop live monitoring."""
        self.is_monitoring = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Live speech monitor stopped")

    def get_live_status(self) -> dict:
        """Get current monitoring status and detected terms."""
        return {
            'is_monitoring': self.is_monitoring,
            'current_event': self._current_event,
            'detected_terms': dict(self._detected_terms),
            'total_detections': sum(self._detected_terms.values()),
        }

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check for live streams
                live_text = self._check_live_sources()

                if live_text:
                    self._analyze_live_text(live_text)

                # Check event schedule for live status
                self._check_scheduled_events()

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(15)  # Check every 15 seconds

    def _check_live_sources(self) -> Optional[str]:
        """Check all live sources and return any live transcript text."""
        texts = []

        # YouTube live chat / captions
        yt_text = self._check_youtube_live()
        if yt_text:
            texts.append(yt_text)

        # C-SPAN live
        cspan_text = self._check_cspan_live()
        if cspan_text:
            texts.append(cspan_text)

        # White House live
        wh_text = self._check_whitehouse_live()
        if wh_text:
            texts.append(wh_text)

        return ' '.join(texts) if texts else None

    def _has_upcoming_event(self, within_minutes: int = 30) -> bool:
        """Check if any scheduled TrumpEvent starts within the given window."""
        now = datetime.utcnow()
        from datetime import timedelta
        window = now + timedelta(minutes=within_minutes)
        try:
            with get_session() as session:
                count = session.query(TrumpEvent).filter(
                    TrumpEvent.start_time.isnot(None),
                    TrumpEvent.start_time <= window,
                    TrumpEvent.start_time >= now - timedelta(hours=3),
                ).count()
                return count > 0
        except Exception:
            return False

    def _check_youtube_live(self) -> Optional[str]:
        """Check YouTube for live Trump streams.

        Rate-limited: only calls the YouTube search API once per 5 minutes,
        and only when a scheduled event is within 30 minutes (or already
        underway). This keeps usage well under the 10,000 daily quota.
        """
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            return None

        # Gate: skip unless an event is near
        if not self._has_upcoming_event(within_minutes=30):
            return None

        # Rate-limit: at most once per 5 minutes (vs every 15s before)
        now_ts = time.time()
        if now_ts - self._last_yt_search < 300:
            return None
        self._last_yt_search = now_ts

        try:
            search_url = 'https://www.googleapis.com/youtube/v3/search'
            params = {
                'key': api_key,
                'q': 'trump live',
                'type': 'video',
                'eventType': 'live',
                'part': 'snippet',
                'maxResults': 5,
            }

            resp = self.session.get(search_url, params=params, timeout=10)
            data = resp.json()

            for item in data.get('items', []):
                title = item['snippet']['title'].lower()
                if any(kw in title for kw in ['trump', 'president', 'white house', 'potus']):
                    self._current_event = {
                        'source': 'youtube',
                        'title': item['snippet']['title'],
                        'video_id': item['id']['videoId'],
                        'started_at': datetime.utcnow().isoformat(),
                    }

                    # Try to get live captions
                    video_id = item['id']['videoId']
                    captions = self._get_live_captions(video_id)
                    if captions:
                        return captions

        except Exception as e:
            logger.debug(f"YouTube live check error: {e}")

        return None

    def _get_live_captions(self, video_id: str) -> Optional[str]:
        """Attempt to get live auto-captions from a YouTube stream."""
        try:
            # Try the timedtext API for live captions
            url = f'https://www.youtube.com/watch?v={video_id}'
            resp = self.session.get(url, timeout=10)

            # Look for caption track info in page source
            import re
            caption_match = re.search(
                r'"captionTracks":\[(.+?)\]', resp.text
            )
            if caption_match:
                import json
                tracks = json.loads(f'[{caption_match.group(1)}]')
                for track in tracks:
                    if 'baseUrl' in track:
                        cap_resp = self.session.get(track['baseUrl'], timeout=10)
                        soup = BeautifulSoup(cap_resp.text, 'html.parser')
                        return soup.get_text(' ', strip=True)

        except Exception as e:
            logger.debug(f"Live caption error: {e}")

        return None

    def _check_cspan_live(self) -> Optional[str]:
        """Check C-SPAN for live Trump coverage."""
        try:
            resp = self.session.get('https://www.c-span.org/', timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')

            live_items = soup.select('.live-item, .now-on, [class*="live"]')
            for item in live_items:
                text = item.get_text(' ', strip=True).lower()
                if any(kw in text for kw in ['trump', 'president', 'white house']):
                    self._current_event = {
                        'source': 'cspan',
                        'title': item.get_text(strip=True),
                        'started_at': datetime.utcnow().isoformat(),
                    }
                    return text

        except Exception as e:
            logger.debug(f"C-SPAN live check error: {e}")

        return None

    def _check_whitehouse_live(self) -> Optional[str]:
        """Check White House for live feed."""
        try:
            resp = self.session.get('https://www.whitehouse.gov/live/', timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')

            live_content = soup.select_one('.live-content, .live-stream, [class*="live"]')
            if live_content:
                text = live_content.get_text(' ', strip=True)
                if text and len(text) > 50:
                    self._current_event = {
                        'source': 'whitehouse',
                        'title': 'White House Live',
                        'started_at': datetime.utcnow().isoformat(),
                    }
                    return text

        except Exception as e:
            logger.debug(f"WH live check error: {e}")

        return None

    def _check_scheduled_events(self):
        """Update live status of scheduled events."""
        now = datetime.utcnow()
        with get_session() as session:
            events = session.query(TrumpEvent).filter(
                TrumpEvent.start_time.isnot(None)
            ).all()

            for event in events:
                if event.start_time and event.start_time <= now:
                    from datetime import timedelta
                    end = event.end_time or (event.start_time + timedelta(hours=3))
                    if now <= end and not event.is_live:
                        event.is_live = True
                        self._notify_event_live(event)
                    elif now > end:
                        event.is_live = False

    def _notify_event_live(self, event: TrumpEvent):
        """Send notification that an event is now live."""
        logger.info(f"EVENT LIVE: {event.title}")
        self._current_event = {
            'source': 'schedule',
            'title': event.title,
            'event_type': event.event_type,
            'started_at': event.start_time.isoformat() if event.start_time else None,
        }

    def _build_term_patterns(self):
        """Pre-compile regex patterns for all tracked terms.

        Rebuilds every 10 minutes to pick up new terms without
        recompiling on every 15-second analysis cycle.
        """
        if time.time() - self._patterns_built_at < 600:
            return  # patterns are fresh enough

        with get_session() as session:
            terms = session.query(Term).all()
            patterns = []
            for term in terms:
                search_terms = [term.normalized_term]
                if term.is_compound and term.sub_terms:
                    search_terms = term.sub_terms
                for st in search_terms:
                    compiled = re.compile(r'\b' + re.escape(st) + r'\b')
                    patterns.append((term.term, compiled))
            self._compiled_patterns = patterns
            self._patterns_built_at = time.time()
            logger.debug(f"Compiled {len(patterns)} term patterns for live monitor")

    def _analyze_live_text(self, text: str):
        """Analyze live text for tracked terms using pre-compiled patterns."""
        self._build_term_patterns()
        text_lower = text.lower()

        for term_name, pattern in self._compiled_patterns:
            matches = pattern.findall(text_lower)
            if matches:
                count = len(matches)
                self._detected_terms[term_name] = (
                    self._detected_terms.get(term_name, 0) + count
                )

                for cb in self._callbacks:
                    try:
                        cb(term_name, count, 'live', datetime.utcnow())
                    except Exception as e:
                        logger.debug(f"Callback error: {e}")
