"""Tracks upcoming Trump public appearances and events."""

import re
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

import requests
from bs4 import BeautifulSoup
import feedparser

from ..database.models import TrumpEvent
from ..database.db import get_session


class EventTracker:
    """Discovers and tracks upcoming Trump events from multiple sources."""

    USER_AGENT = (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.USER_AGENT})

    def update_events(self) -> dict:
        """Scrape all sources for upcoming Trump events."""
        stats = {'new_events': 0, 'sources': {}}

        scrapers = [
            ('whitehouse_schedule', self._scrape_whitehouse_schedule),
            ('factbase_calendar', self._scrape_factbase_calendar),
            ('google_news', self._scrape_google_news_events),
            ('cspan_schedule', self._scrape_cspan_schedule),
        ]

        for name, scraper_fn in scrapers:
            try:
                count = scraper_fn()
                stats['sources'][name] = count
                stats['new_events'] += count
            except Exception as e:
                logger.error(f"[{name}] event scraper failed: {e}")
                stats['sources'][name] = f"error: {e}"

        return stats

    def _save_event(self, title: str, event_type: str,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    location: Optional[str] = None,
                    source_url: Optional[str] = None,
                    notes: Optional[str] = None,
                    topics: Optional[list] = None,
                    is_confirmed: bool = True) -> bool:
        """Save event to database. Returns True if new."""
        with get_session() as session:
            # Check for duplicates by title + date
            if start_time:
                existing = session.query(TrumpEvent).filter(
                    TrumpEvent.title == title,
                    TrumpEvent.start_time == start_time
                ).first()
            else:
                existing = session.query(TrumpEvent).filter(
                    TrumpEvent.title == title
                ).first()

            if existing:
                # Update if we have more info
                if location and not existing.location:
                    existing.location = location
                if source_url and not existing.source_url:
                    existing.source_url = source_url
                return False

            event = TrumpEvent(
                title=title,
                event_type=event_type,
                location=location,
                start_time=start_time,
                end_time=end_time,
                is_confirmed=is_confirmed,
                source_url=source_url,
                notes=notes,
                topics=topics,
            )
            session.add(event)
            return True

    def _scrape_whitehouse_schedule(self) -> int:
        """Scrape the White House public schedule."""
        count = 0
        try:
            url = 'https://www.whitehouse.gov/schedule/'
            resp = self.session.get(url, timeout=30)
            soup = BeautifulSoup(resp.text, 'html.parser')

            for item in soup.select('.schedule-item, .event-item, article'):
                try:
                    title_el = item.select_one('h2, h3, .title, .event-title')
                    if not title_el:
                        continue

                    title = title_el.get_text(strip=True)
                    if not any(kw in title.lower() for kw in
                               ['president', 'trump', 'remarks', 'press', 'meeting', 'briefing']):
                        continue

                    time_el = item.select_one('time, .time, .event-time')
                    start_time = None
                    if time_el:
                        dt_str = time_el.get('datetime') or time_el.get_text(strip=True)
                        try:
                            start_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            pass

                    location_el = item.select_one('.location, .event-location')
                    location = location_el.get_text(strip=True) if location_el else None

                    event_type = self._classify_event(title)

                    if self._save_event(
                        title=title,
                        event_type=event_type,
                        start_time=start_time,
                        location=location,
                        source_url=url,
                    ):
                        count += 1

                except Exception as e:
                    logger.debug(f"Error parsing WH schedule item: {e}")

        except Exception as e:
            logger.warning(f"WH schedule scraper error: {e}")

        return count

    def _scrape_factbase_calendar(self) -> int:
        """Scrape Factba.se calendar for Trump events."""
        count = 0
        try:
            url = 'https://factba.se/biden-trump/calendar'
            resp = self.session.get(url, timeout=30)
            soup = BeautifulSoup(resp.text, 'html.parser')

            for item in soup.select('.calendar-event, .event-item, .fc-event'):
                try:
                    title = item.get_text(strip=True)
                    href = item.get('href', '')
                    start_str = item.get('data-start', '')

                    start_time = None
                    if start_str:
                        try:
                            start_time = datetime.fromisoformat(start_str)
                        except ValueError:
                            pass

                    if self._save_event(
                        title=title,
                        event_type=self._classify_event(title),
                        start_time=start_time,
                        source_url=href if href.startswith('http') else None,
                    ):
                        count += 1

                except Exception as e:
                    logger.debug(f"Error parsing factbase event: {e}")

        except Exception as e:
            logger.warning(f"Factbase calendar error: {e}")

        return count

    def _scrape_google_news_events(self) -> int:
        """Search Google News RSS for upcoming Trump events."""
        count = 0
        queries = [
            'trump rally schedule',
            'trump press conference today',
            'trump speech today',
            'trump public appearance schedule',
        ]

        for query in queries:
            try:
                rss_url = f'https://news.google.com/rss/search?q={query.replace(" ", "+")}&hl=en-US&gl=US&ceid=US:en'
                feed = feedparser.parse(rss_url)

                for entry in feed.entries[:10]:
                    title = entry.get('title', '')
                    # Look for event indicators
                    if not any(kw in title.lower() for kw in
                               ['rally', 'speech', 'press conference', 'appearance',
                                'remarks', 'event', 'town hall', 'signing']):
                        continue

                    published = entry.get('published', '')
                    pub_date = None
                    if published:
                        from dateutil import parser as dateparser
                        try:
                            pub_date = dateparser.parse(published)
                        except (ValueError, TypeError):
                            pass

                    link = entry.get('link', '')

                    if self._save_event(
                        title=title,
                        event_type=self._classify_event(title),
                        start_time=pub_date,
                        source_url=link,
                        is_confirmed=False,
                        notes='Discovered via Google News - confirm timing',
                    ):
                        count += 1

            except Exception as e:
                logger.debug(f"Google News RSS error for '{query}': {e}")

        return count

    def _scrape_cspan_schedule(self) -> int:
        """Scrape C-SPAN schedule for Trump appearances."""
        count = 0
        try:
            url = 'https://www.c-span.org/schedule/'
            resp = self.session.get(url, timeout=30)
            soup = BeautifulSoup(resp.text, 'html.parser')

            for item in soup.select('.schedule-item, tr'):
                try:
                    text = item.get_text(' ', strip=True)
                    if 'trump' not in text.lower() and 'president' not in text.lower():
                        continue

                    title_el = item.select_one('a, .title')
                    if not title_el:
                        continue

                    title = title_el.get_text(strip=True)
                    href = title_el.get('href', '')

                    time_el = item.select_one('.time, time')
                    start_time = None
                    if time_el:
                        try:
                            start_time = datetime.fromisoformat(
                                time_el.get('datetime', '').replace('Z', '+00:00')
                            )
                        except (ValueError, AttributeError):
                            pass

                    if self._save_event(
                        title=title,
                        event_type='cspan_appearance',
                        start_time=start_time,
                        source_url=f'https://www.c-span.org{href}' if not href.startswith('http') else href,
                    ):
                        count += 1

                except Exception as e:
                    logger.debug(f"Error parsing CSPAN schedule item: {e}")

        except Exception as e:
            logger.warning(f"CSPAN schedule error: {e}")

        return count

    def _classify_event(self, title: str) -> str:
        """Classify event type from title."""
        title_lower = title.lower()
        if 'rally' in title_lower or 'maga' in title_lower:
            return 'rally'
        if 'press conference' in title_lower or 'presser' in title_lower:
            return 'press_conference'
        if 'interview' in title_lower:
            return 'interview'
        if 'signing' in title_lower or 'executive order' in title_lower:
            return 'signing'
        if 'briefing' in title_lower:
            return 'briefing'
        if 'meeting' in title_lower:
            return 'meeting'
        if 'dinner' in title_lower or 'gala' in title_lower:
            return 'dinner'
        if 'town hall' in title_lower:
            return 'town_hall'
        if 'debate' in title_lower:
            return 'debate'
        return 'appearance'

    def get_upcoming_events(self, days: int = 30) -> list[dict]:
        """Get upcoming events within the next N days."""
        cutoff = datetime.utcnow() + timedelta(days=days)

        with get_session() as session:
            events = session.query(TrumpEvent).filter(
                TrumpEvent.start_time >= datetime.utcnow(),
                TrumpEvent.start_time <= cutoff
            ).order_by(TrumpEvent.start_time).all()

            return [
                {
                    'id': e.id,
                    'title': e.title,
                    'event_type': e.event_type,
                    'location': e.location,
                    'start_time': e.start_time.isoformat() if e.start_time else None,
                    'end_time': e.end_time.isoformat() if e.end_time else None,
                    'is_live': e.is_live,
                    'is_confirmed': e.is_confirmed,
                    'source_url': e.source_url,
                    'topics': e.topics,
                }
                for e in events
            ]

    def get_live_events(self) -> list[dict]:
        """Get events that are currently live."""
        with get_session() as session:
            events = session.query(TrumpEvent).filter_by(is_live=True).all()
            return [
                {
                    'id': e.id,
                    'title': e.title,
                    'event_type': e.event_type,
                    'start_time': e.start_time.isoformat() if e.start_time else None,
                }
                for e in events
            ]

    def check_and_update_live_status(self):
        """Update is_live flag based on current time and event schedule."""
        now = datetime.utcnow()

        with get_session() as session:
            events = session.query(TrumpEvent).filter(
                TrumpEvent.start_time.isnot(None)
            ).all()

            for event in events:
                end_time = event.end_time or (event.start_time + timedelta(hours=2))
                was_live = event.is_live

                if event.start_time <= now <= end_time:
                    event.is_live = True
                    if not was_live:
                        logger.info(f"EVENT NOW LIVE: {event.title}")
                else:
                    event.is_live = False
