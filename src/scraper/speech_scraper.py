"""Multi-source scraper for Trump speeches and public appearances."""

import re
import json
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

import requests
from bs4 import BeautifulSoup
import feedparser

from ..database.models import Speech
from ..database.db import get_session


class SpeechScraper:
    """Scrapes Trump speeches from multiple sources."""

    USER_AGENT = (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.USER_AGENT})

    def scrape_all_sources(self) -> dict:
        """Run all scrapers and return summary."""
        stats = {'total_new': 0, 'sources': {}}

        scrapers = [
            ('whitehouse', self.scrape_whitehouse_remarks),
            ('rev_transcripts', self.scrape_rev_transcripts),
            ('factba_se', self.scrape_factbase),
            ('cspan', self.scrape_cspan),
            ('miller_center', self.scrape_miller_center),
            ('youtube_transcripts', self.scrape_youtube_channels),
        ]

        for name, scraper_fn in scrapers:
            try:
                count = scraper_fn()
                stats['sources'][name] = count
                stats['total_new'] += count
                logger.info(f"[{name}] scraped {count} new speeches")
            except Exception as e:
                logger.error(f"[{name}] scraper failed: {e}")
                stats['sources'][name] = f"error: {e}"

        return stats

    def _save_speech(self, source: str, source_id: str, title: str,
                     date: datetime, transcript: Optional[str] = None,
                     source_url: Optional[str] = None,
                     speech_type: Optional[str] = None,
                     duration: Optional[int] = None,
                     metadata: Optional[dict] = None) -> bool:
        """Save a speech to database if not already present. Returns True if new."""
        with get_session() as session:
            existing = session.query(Speech).filter_by(
                source=source, source_id=source_id
            ).first()
            if existing:
                # Update transcript if we now have one
                if transcript and not existing.transcript:
                    existing.transcript = transcript
                    existing.word_count = len(transcript.split()) if transcript else 0
                    logger.debug(f"Updated transcript for {source}/{source_id}")
                return False

            speech = Speech(
                source=source,
                source_id=source_id,
                source_url=source_url,
                title=title,
                speech_type=speech_type or self._classify_speech_type(title),
                date=date,
                duration_seconds=duration,
                transcript=transcript,
                transcript_source=source if transcript else None,
                word_count=len(transcript.split()) if transcript else 0,
                is_processed=False,
                raw_metadata=metadata,
            )
            session.add(speech)
            return True

    def _classify_speech_type(self, title: str) -> str:
        """Classify speech type from title."""
        title_lower = title.lower()
        if any(w in title_lower for w in ['rally', 'maga']):
            return 'rally'
        if any(w in title_lower for w in ['press conference', 'presser', 'press briefing']):
            return 'press_conference'
        if any(w in title_lower for w in ['interview', 'fox', 'cnn', 'msnbc', 'newsmax']):
            return 'interview'
        if any(w in title_lower for w in ['state of the union', 'sotu', 'address to congress']):
            return 'address'
        if any(w in title_lower for w in ['remarks', 'speech', 'statement']):
            return 'remarks'
        if any(w in title_lower for w in ['debate']):
            return 'debate'
        if any(w in title_lower for w in ['town hall']):
            return 'town_hall'
        if any(w in title_lower for w in ['signing', 'executive order']):
            return 'signing'
        return 'other'

    # --- Source: White House ---

    def scrape_whitehouse_remarks(self) -> int:
        """Scrape remarks and speeches from whitehouse.gov."""
        count = 0
        base_url = 'https://www.whitehouse.gov/briefing-room/speeches-remarks/'

        for page in range(1, 20):
            try:
                url = f'{base_url}page/{page}/' if page > 1 else base_url
                resp = self.session.get(url, timeout=30)
                if resp.status_code != 200:
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')
                articles = soup.select('article, .news-item, .briefing-statement')

                if not articles:
                    # Try alternate selectors
                    articles = soup.select('li.news-item, div.views-row')

                for article in articles:
                    try:
                        link_tag = article.select_one('a[href]')
                        if not link_tag:
                            continue

                        title = link_tag.get_text(strip=True)
                        href = link_tag['href']
                        if not href.startswith('http'):
                            href = f'https://www.whitehouse.gov{href}'

                        # Check if it's Trump-related
                        if not any(kw in title.lower() for kw in ['president', 'trump', 'remarks', 'statement']):
                            continue

                        date_tag = article.select_one('time, .date, .meta-date')
                        date = datetime.now()
                        if date_tag:
                            date_str = date_tag.get('datetime') or date_tag.get_text(strip=True)
                            try:
                                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            except (ValueError, AttributeError):
                                pass

                        source_id = href.split('/')[-2] if href.endswith('/') else href.split('/')[-1]

                        # Fetch full transcript
                        transcript = self._fetch_article_text(href)

                        if self._save_speech(
                            source='whitehouse',
                            source_id=source_id,
                            title=title,
                            date=date,
                            transcript=transcript,
                            source_url=href,
                        ):
                            count += 1

                    except Exception as e:
                        logger.debug(f"Error parsing WH article: {e}")

            except Exception as e:
                logger.warning(f"Error on WH page {page}: {e}")
                break

        return count

    def _fetch_article_text(self, url: str) -> Optional[str]:
        """Fetch and extract main text content from an article URL."""
        try:
            import trafilatura
            resp = self.session.get(url, timeout=30)
            text = trafilatura.extract(resp.text)
            return text
        except ImportError:
            try:
                resp = self.session.get(url, timeout=30)
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Remove scripts, styles
                for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                body = soup.select_one('article, .entry-content, .body-content, main')
                if body:
                    return body.get_text(separator=' ', strip=True)
                return soup.get_text(separator=' ', strip=True)[:50000]
            except Exception:
                return None
        except Exception:
            return None

    # --- Source: Rev.com Transcripts ---

    def scrape_rev_transcripts(self) -> int:
        """Scrape Trump speech transcripts from Rev.com."""
        count = 0
        base_url = 'https://www.rev.com/blog/transcript-category/donald-trump-transcripts'

        for page in range(1, 30):
            try:
                url = f'{base_url}/page/{page}' if page > 1 else base_url
                resp = self.session.get(url, timeout=30)
                if resp.status_code != 200:
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')
                articles = soup.select('article, .fl-post-column')

                for article in articles:
                    try:
                        link = article.select_one('a[href*="transcript"]')
                        if not link:
                            link = article.select_one('a[href]')
                        if not link:
                            continue

                        title = link.get_text(strip=True)
                        href = link['href']
                        source_id = href.rstrip('/').split('/')[-1]

                        # Parse date from title or meta
                        date = self._extract_date_from_text(title)

                        transcript = self._fetch_article_text(href)

                        if self._save_speech(
                            source='rev_transcripts',
                            source_id=source_id,
                            title=title,
                            date=date,
                            transcript=transcript,
                            source_url=href,
                        ):
                            count += 1

                    except Exception as e:
                        logger.debug(f"Error parsing Rev article: {e}")

            except Exception as e:
                logger.warning(f"Error on Rev page {page}: {e}")
                break

        return count

    # --- Source: Factba.se ---

    def scrape_factbase(self) -> int:
        """Scrape from Factba.se (comprehensive Trump transcript archive)."""
        count = 0
        try:
            # Factba.se has an API-like interface
            api_url = 'https://factba.se/json/json-transcript.php'
            params = {'q': '', 'f': 'trump', 'dt': 'speech', 'p': 1}

            for page in range(1, 50):
                params['p'] = page
                resp = self.session.get(api_url, params=params, timeout=30)
                if resp.status_code != 200:
                    break

                try:
                    data = resp.json()
                except (json.JSONDecodeError, ValueError):
                    # Try scraping HTML instead
                    count += self._scrape_factbase_html(page)
                    continue

                if not data:
                    break

                for item in data:
                    try:
                        source_id = str(item.get('id', item.get('slug', '')))
                        title = item.get('title', '')
                        date_str = item.get('date', '')
                        transcript = item.get('text', '')

                        date = datetime.now()
                        if date_str:
                            try:
                                date = datetime.fromisoformat(date_str)
                            except ValueError:
                                pass

                        if self._save_speech(
                            source='factbase',
                            source_id=source_id,
                            title=title,
                            date=date,
                            transcript=transcript,
                            source_url=f'https://factba.se/transcript/{source_id}',
                            metadata=item,
                        ):
                            count += 1

                    except Exception as e:
                        logger.debug(f"Error parsing factbase item: {e}")

        except Exception as e:
            logger.warning(f"Factbase scraper error: {e}")

        return count

    def _scrape_factbase_html(self, page: int) -> int:
        """Fallback HTML scraping for Factba.se."""
        count = 0
        try:
            url = f'https://factba.se/biden-trump/transcript?page={page}'
            resp = self.session.get(url, timeout=30)
            soup = BeautifulSoup(resp.text, 'html.parser')

            for item in soup.select('.transcript-listing, .topic-item'):
                link = item.select_one('a[href]')
                if not link:
                    continue
                title = link.get_text(strip=True)
                href = link['href']
                source_id = href.rstrip('/').split('/')[-1]
                date = self._extract_date_from_text(title)

                if self._save_speech(
                    source='factbase',
                    source_id=source_id,
                    title=title,
                    date=date,
                    source_url=href if href.startswith('http') else f'https://factba.se{href}',
                ):
                    count += 1

        except Exception as e:
            logger.debug(f"Factbase HTML scrape page {page} error: {e}")

        return count

    # --- Source: C-SPAN ---

    def scrape_cspan(self) -> int:
        """Scrape Trump appearances from C-SPAN."""
        count = 0
        try:
            search_url = 'https://www.c-span.org/search/'
            params = {'query': 'trump', 'sort': 'Most+Recent'}

            resp = self.session.get(search_url, params=params, timeout=30)
            soup = BeautifulSoup(resp.text, 'html.parser')

            for item in soup.select('.video-result, .search-result'):
                try:
                    link = item.select_one('a[href]')
                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    href = link['href']
                    if not href.startswith('http'):
                        href = f'https://www.c-span.org{href}'

                    source_id = href.rstrip('/').split('/')[-1]
                    date = self._extract_date_from_text(
                        item.get_text(' ', strip=True)
                    )

                    if self._save_speech(
                        source='cspan',
                        source_id=source_id,
                        title=title,
                        date=date,
                        source_url=href,
                        speech_type='cspan_appearance',
                    ):
                        count += 1

                except Exception as e:
                    logger.debug(f"Error parsing CSPAN item: {e}")

        except Exception as e:
            logger.warning(f"CSPAN scraper error: {e}")

        return count

    # --- Source: Miller Center (UVA) ---

    def scrape_miller_center(self) -> int:
        """Scrape presidential speeches from Miller Center at UVA."""
        count = 0
        try:
            url = 'https://millercenter.org/the-presidency/presidential-speeches'
            params = {'field_president_target_id': '278'}  # Trump's ID on the site

            resp = self.session.get(url, params=params, timeout=30)
            soup = BeautifulSoup(resp.text, 'html.parser')

            for item in soup.select('.views-row, .speech-listing'):
                try:
                    link = item.select_one('a[href]')
                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    href = link['href']
                    if not href.startswith('http'):
                        href = f'https://millercenter.org{href}'

                    source_id = href.rstrip('/').split('/')[-1]
                    date = self._extract_date_from_text(
                        item.get_text(' ', strip=True)
                    )

                    # Fetch transcript
                    transcript = self._fetch_article_text(href)

                    if self._save_speech(
                        source='miller_center',
                        source_id=source_id,
                        title=title,
                        date=date,
                        transcript=transcript,
                        source_url=href,
                    ):
                        count += 1

                except Exception as e:
                    logger.debug(f"Error parsing Miller Center item: {e}")

        except Exception as e:
            logger.warning(f"Miller Center scraper error: {e}")

        return count

    # --- Source: YouTube Channels ---

    def scrape_youtube_channels(self) -> int:
        """Scrape transcripts from key YouTube channels."""
        import os

        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            logger.warning("No YouTube API key configured, skipping YouTube scraper")
            return 0

        count = 0
        # Key channels that frequently post Trump speeches
        channels = [
            'UCBi2mrWuNuyYy4gbM6fU18Q',  # White House
            'UC_DqXiifkgFBYVkNOgMSawg',  # RSBN (Right Side Broadcasting)
            'UCaXkIU1QidjPwiAYu6GcHjg',  # CSPAN
            'UCupvZG-5ko_eiXAupbDfxWw',  # CNN
            'UCeY0bbntWzzVIaj2z3QigXg',  # NBC
        ]

        for channel_id in channels:
            try:
                search_url = 'https://www.googleapis.com/youtube/v3/search'
                params = {
                    'key': api_key,
                    'channelId': channel_id,
                    'q': 'trump speech OR trump remarks OR trump rally OR trump press conference',
                    'type': 'video',
                    'order': 'date',
                    'maxResults': 50,
                    'part': 'snippet',
                }

                resp = self.session.get(search_url, params=params, timeout=30)
                data = resp.json()

                for item in data.get('items', []):
                    try:
                        video_id = item['id']['videoId']
                        snippet = item['snippet']
                        title = snippet['title']
                        date_str = snippet['publishedAt']
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                        # Try to get transcript via YouTube
                        transcript = self._get_youtube_transcript(video_id)

                        if self._save_speech(
                            source='youtube',
                            source_id=video_id,
                            title=title,
                            date=date,
                            transcript=transcript,
                            source_url=f'https://www.youtube.com/watch?v={video_id}',
                            metadata=snippet,
                        ):
                            count += 1

                    except Exception as e:
                        logger.debug(f"Error processing YT video: {e}")

            except Exception as e:
                logger.debug(f"Error searching YT channel {channel_id}: {e}")

        return count

    def _get_youtube_transcript(self, video_id: str) -> Optional[str]:
        """Attempt to get a YouTube video transcript."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join(item['text'] for item in transcript_list)
        except Exception:
            return None

    def _extract_date_from_text(self, text: str) -> datetime:
        """Try to extract a date from text, fallback to now."""
        import re
        from dateutil import parser as dateparser

        # Common date patterns
        patterns = [
            r'(\w+ \d{1,2},? \d{4})',
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
        ]

        for pat in patterns:
            match = re.search(pat, text)
            if match:
                try:
                    return dateparser.parse(match.group(1))
                except (ValueError, TypeError):
                    continue

        return datetime.now()
