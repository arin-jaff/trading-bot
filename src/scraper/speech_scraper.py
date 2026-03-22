"""Multi-source scraper for Trump speeches and public appearances.

Sources (updated June 2025):
  1. Rev.com transcripts  - /category/donald-trump  (WORKING)
  2. Google News RSS       - aggregates transcript links (NEW)
  3. White House /remarks/  - whitehouse.gov (UPDATED URL)
  4. Roll Call / Factbase   - rollcall.com/factbase (REPLACES dead factba.se)
  5. C-SPAN video search    - c-span.org (UPDATED selectors)
  6. YouTube channels       - via Data API + transcript API (WORKING)
"""

import os
import re
import json
import time
from datetime import datetime, timedelta
from typing import Optional, List
from urllib.parse import urljoin, urlparse, parse_qs
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
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    )

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def scrape_all_sources(self) -> dict:
        """Run all scrapers and return summary."""
        stats = {'total_new': 0, 'sources': {}}

        scrapers = [
            ('rev_transcripts', self.scrape_rev_transcripts),
            ('google_news_rss', self.scrape_google_news_rss),
            ('whitehouse', self.scrape_whitehouse_remarks),
            ('rollcall_factbase', self.scrape_rollcall_factbase),
            ('cspan', self.scrape_cspan),
            ('cspan_transcripts', self.scrape_cspan_transcripts),
            ('youtube_transcripts', self.scrape_youtube_channels),
            ('youtube_yt_dlp', self.scrape_youtube_yt_dlp),
            ('presidency_project', self.scrape_presidency_project),
            ('twitter_archive', self.scrape_trump_twitter_archive),
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

    # ------------------------------------------------------------------
    # Database helper
    # ------------------------------------------------------------------

    def _save_speech(self, source: str, source_id: str, title: str,
                     date: datetime, transcript: Optional[str] = None,
                     source_url: Optional[str] = None,
                     speech_type: Optional[str] = None,
                     duration: Optional[int] = None,
                     metadata: Optional[dict] = None) -> bool:
        """Save a speech to database if not already present. Returns True if new.

        4C: Also checks for duplicate transcripts via content hash to prevent
        the same speech scraped from multiple sources from inflating frequencies.
        """
        with get_session() as session:
            existing = session.query(Speech).filter_by(
                source=source, source_id=source_id
            ).first()
            if existing:
                if transcript and not existing.transcript:
                    existing.transcript = transcript
                    existing.word_count = len(transcript.split()) if transcript else 0
                    logger.debug(f"Updated transcript for {source}/{source_id}")
                return False

            # 4C: Duplicate transcript detection — check first 200 chars
            # against existing transcripts using SQL LIKE for efficiency
            if transcript and len(transcript) > 50:
                normalized_prefix = ' '.join(transcript.lower().split())[:200]
                # Check a shorter prefix via SQL to avoid full scan
                search_prefix = normalized_prefix[:80]
                from sqlalchemy import func as sa_func
                dupe_count = session.query(Speech).filter(
                    Speech.transcript.isnot(None),
                    sa_func.lower(sa_func.substr(Speech.transcript, 1, 80)).contains(
                        search_prefix[:40]
                    ),
                    Speech.source != source,  # allow same source to update
                ).count()
                if dupe_count > 0:
                    logger.debug(
                        f"4C: Likely duplicate transcript — "
                        f"'{title[:50]}' matches {dupe_count} existing speech(es)"
                    )
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
        t = title.lower()
        if any(w in t for w in ['rally', 'maga']):
            return 'rally'
        if any(w in t for w in ['press conference', 'presser', 'press briefing']):
            return 'press_conference'
        if any(w in t for w in ['interview', 'fox', 'cnn', 'msnbc', 'newsmax']):
            return 'interview'
        if any(w in t for w in ['state of the union', 'sotu', 'address to congress']):
            return 'address'
        if any(w in t for w in ['remarks', 'speech', 'statement']):
            return 'remarks'
        if 'debate' in t:
            return 'debate'
        if 'town hall' in t:
            return 'town_hall'
        if any(w in t for w in ['signing', 'executive order']):
            return 'signing'
        return 'other'

    # ------------------------------------------------------------------
    # Text extraction helpers
    # ------------------------------------------------------------------

    def _fetch_article_text(self, url: str) -> Optional[str]:
        """Fetch and extract main text content from an article URL."""
        try:
            resp = self.session.get(url, timeout=30, allow_redirects=True)
            if resp.status_code != 200:
                return None
        except Exception:
            return None

        # Try trafilatura first (best quality)
        try:
            import trafilatura
            text = trafilatura.extract(resp.text, include_comments=False)
            if text and len(text) > 200:
                return text
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: BeautifulSoup
        try:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            body = soup.select_one(
                'article, .entry-content, .body-content, '
                '.transcript-content, .fl-module-content, '
                '.fl-callout-text, .post-content, main, [role="main"]'
            )
            if body:
                return body.get_text(separator=' ', strip=True)
            return soup.get_text(separator=' ', strip=True)[:50000]
        except Exception:
            return None

    def _get_youtube_transcript(self, video_id: str) -> Optional[str]:
        """Attempt to get a YouTube video transcript.

        Supports both the new youtube-transcript-api (>=1.0, .fetch())
        and the legacy API (.get_transcript()) for backwards compat.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id)
            return ' '.join(snippet.text for snippet in transcript)
        except AttributeError:
            # Legacy API (<1.0)
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                return ' '.join(item['text'] for item in transcript_list)
            except Exception:
                return None
        except Exception:
            return None

    def _extract_date_from_text(self, text: str) -> datetime:
        """Try to extract a date from text, fallback to now."""
        from dateutil import parser as dateparser

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

    def _resolve_google_news_url(self, google_url: str) -> str:
        """Follow Google News redirect to get the real article URL."""
        try:
            resp = self.session.head(google_url, allow_redirects=True, timeout=15)
            return resp.url
        except Exception:
            try:
                resp = self.session.get(google_url, allow_redirects=True, timeout=15)
                return resp.url
            except Exception:
                return google_url

    # ==================================================================
    # SOURCE 1: Rev.com Transcripts
    # ==================================================================

    def scrape_rev_transcripts(self) -> int:
        """Scrape Trump speech transcripts from Rev.com.

        Rev.com transcript category:
          - Category page: /blog/transcript-category/donald-trump-transcripts
          - Individual transcripts: /transcripts/<slug>
          - Pagination: /page/N/
        """
        count = 0
        base_url = 'https://www.rev.com/blog/transcript-category/donald-trump-transcripts'

        for page in range(1, 30):
            try:
                if page == 1:
                    url = base_url
                else:
                    url = f'{base_url}/page/{page}/'

                resp = self.session.get(url, timeout=30)
                if resp.status_code != 200:
                    logger.debug(f"Rev.com page {page} returned {resp.status_code}")
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')

                # Find all links pointing to /transcripts/
                links = soup.find_all('a', href=True)
                transcript_links = []
                for link in links:
                    href = link['href']
                    if '/transcripts/' in href:
                        full_url = href if href.startswith('http') else f'https://www.rev.com{href}'
                        title = link.get_text(strip=True)
                        if title and len(title) > 5:
                            transcript_links.append((title, full_url))

                # Deduplicate by URL
                seen_urls = set()
                unique_links = []
                for title, url in transcript_links:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_links.append((title, url))

                if not unique_links:
                    logger.debug(f"Rev.com page {page}: no transcript links found, stopping")
                    break

                for title, href in unique_links:
                    try:
                        source_id = href.rstrip('/').split('/')[-1]

                        # Filter for Trump-related transcripts
                        title_lower = title.lower()
                        if not any(kw in title_lower for kw in [
                            'trump', 'president', 'state of the union',
                            'press conference', 'rally', 'remarks',
                            'white house', 'oval office', 'executive order',
                            'medal', 'address', 'cabinet', 'bilateral',
                        ]):
                            continue

                        date = self._extract_date_from_text(title)
                        transcript = self._fetch_article_text(href)
                        time.sleep(2)  # Rate limit for Rev.com

                        if self._save_speech(
                            source='rev_transcripts',
                            source_id=source_id,
                            title=title,
                            date=date,
                            transcript=transcript,
                            source_url=href,
                        ):
                            count += 1
                            logger.debug(f"Rev.com: saved '{title[:60]}'")

                    except Exception as e:
                        logger.debug(f"Error parsing Rev article: {e}")

            except Exception as e:
                logger.warning(f"Error on Rev page {page}: {e}")
                break

            time.sleep(2)  # Rate limit between pages

        return count

    # ==================================================================
    # SOURCE 2: Google News RSS (aggregator - finds transcripts anywhere)
    # ==================================================================

    def scrape_google_news_rss(self) -> int:
        """Scrape transcript links from Google News RSS feeds.

        This is a meta-source that discovers transcripts published on
        Rev.com, PBS, AP News, ABC, Roll Call, C-SPAN, and other outlets.
        """
        count = 0
        queries = [
            'trump speech transcript',
            'trump remarks transcript',
            'trump press conference transcript',
            'trump rally transcript full',
            'president trump address transcript',
        ]

        seen_urls = set()

        for query in queries:
            try:
                rss_url = f'https://news.google.com/rss/search?q={query.replace(" ", "+")}&hl=en-US&gl=US&ceid=US:en'
                feed = feedparser.parse(rss_url)

                for entry in feed.entries:
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        published = entry.get('published', '')

                        if not title or not link:
                            continue

                        # Only process entries that are likely transcripts
                        title_lower = title.lower()
                        if not any(kw in title_lower for kw in [
                            'transcript', 'full text', 'full speech',
                            'remarks', 'read:', 'full remarks',
                        ]):
                            continue

                        # Resolve Google News redirect to actual URL
                        real_url = self._resolve_google_news_url(link)

                        if real_url in seen_urls:
                            continue
                        seen_urls.add(real_url)

                        # Build source_id from the real URL
                        source_id = urlparse(real_url).path.rstrip('/').split('/')[-1]
                        if not source_id or len(source_id) < 3:
                            source_id = re.sub(r'[^a-z0-9]+', '-', title.lower())[:80]

                        # Parse date
                        date = datetime.now()
                        if published:
                            try:
                                from dateutil import parser as dateparser
                                date = dateparser.parse(published)
                            except Exception:
                                pass

                        # Fetch the actual transcript text
                        transcript = self._fetch_article_text(real_url)

                        if self._save_speech(
                            source='google_news_rss',
                            source_id=source_id,
                            title=title,
                            date=date,
                            transcript=transcript,
                            source_url=real_url,
                        ):
                            count += 1
                            logger.debug(f"Google News: saved '{title[:60]}'")

                    except Exception as e:
                        logger.debug(f"Error parsing Google News entry: {e}")

            except Exception as e:
                logger.warning(f"Google News RSS error for '{query}': {e}")

        return count

    # ==================================================================
    # SOURCE 3: White House Remarks
    # ==================================================================

    def scrape_whitehouse_remarks(self) -> int:
        """Scrape remarks from whitehouse.gov.

        The WH site now uses:
          - Listing page: /remarks/  (not /briefing-room/speeches-remarks/)
          - Pagination: ?query-10-page=N
          - Remark detail pages: /remarks/YYYY/MM/slug/
          - Video pages: /videos/slug/  (most content lives here)
          - Title links via h2 a, h3 a, .wp-block-post-title a
        """
        count = 0
        base_url = 'https://www.whitehouse.gov/remarks/'

        for page in range(1, 90):
            try:
                if page == 1:
                    url = base_url
                else:
                    url = f'{base_url}?query-10-page={page}'

                resp = self.session.get(url, timeout=30)
                if resp.status_code != 200:
                    logger.debug(f"WH page {page} returned {resp.status_code}")
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')

                # Collect links from title elements (h2 a, h3 a, .wp-block-post-title a)
                page_links = []
                for el in soup.select('h2 a[href], h3 a[href], .wp-block-post-title a[href]'):
                    href = el['href']
                    title = el.get_text(strip=True)
                    if not title or len(title) < 5:
                        continue
                    # Accept /remarks/YYYY/... and /videos/... links
                    if '/remarks/' in href or '/videos/' in href:
                        full_url = href if href.startswith('http') else f'https://www.whitehouse.gov{href}'
                        page_links.append((title, full_url))

                # Deduplicate
                seen = set()
                unique_links = []
                for title, link_url in page_links:
                    if link_url not in seen:
                        seen.add(link_url)
                        unique_links.append((title, link_url))

                if not unique_links:
                    logger.debug(f"WH page {page}: no remark links found, stopping")
                    break

                for title, href in unique_links:
                    try:
                        source_id = href.rstrip('/').split('/')[-1]
                        if not source_id:
                            source_id = '-'.join(href.rstrip('/').split('/')[-3:])

                        # Try date from URL first (/remarks/2025/01/slug/)
                        date = None
                        date_match = re.search(r'/(\d{4})/(\d{2})(?:/(\d{2}))?/', href)
                        if date_match:
                            try:
                                day = int(date_match.group(3)) if date_match.group(3) else 1
                                date = datetime(
                                    int(date_match.group(1)),
                                    int(date_match.group(2)),
                                    day,
                                )
                            except ValueError:
                                pass
                        if date is None:
                            date = self._extract_date_from_text(title)

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
                            logger.debug(f"WH: saved '{title[:60]}'")

                    except Exception as e:
                        logger.debug(f"Error parsing WH article: {e}")

            except Exception as e:
                logger.warning(f"Error on WH page {page}: {e}")
                break

        return count

    # ==================================================================
    # SOURCE 4: Roll Call / Factbase
    # ==================================================================

    def scrape_rollcall_factbase(self) -> int:
        """Scrape from rollcall.com/factbase (replaces dead factba.se).

        NOTE: Roll Call/Factbase does not host actual transcripts — the
        search returns news articles *about* Trump, which pollute the
        training corpus.  The original factba.se transcript database was
        not migrated to rollcall.com.  Disabled to prevent corpus
        contamination.  Transcript coverage comes from Rev.com, White
        House, Presidency Project, and Google News RSS instead.
        """
        logger.debug("Roll Call/Factbase disabled — returns news articles, not transcripts")
        return 0

    # ==================================================================
    # SOURCE 5: C-SPAN
    # ==================================================================

    def scrape_cspan(self) -> int:
        """Scrape Trump appearances from C-SPAN search.

        NOTE: C-SPAN search results are now fully JS-rendered — no video
        links appear in the raw HTML.  C-SPAN content is still picked up
        by the Google News RSS scraper (source #2), so this is not a gap.
        Keeping the method as a stub to avoid breaking scrape_all_sources().
        """
        logger.debug("C-SPAN search is JS-rendered; coverage via Google News RSS")
        return 0

    # ==================================================================
    # SOURCE 6: YouTube Channels
    # ==================================================================

    def scrape_youtube_channels(self) -> int:
        """Scrape transcripts from key YouTube channels via Data API."""
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            logger.warning("No YOUTUBE_API_KEY configured, skipping YouTube scraper")
            return 0

        count = 0
        channels = [
            'UCYxRlFDqcWM4y7FfpiAN3KQ',  # The White House
            'UC_DqXiifkgFBYVkNOgMSawg',  # RSBN (Right Side Broadcasting)
            'UCaXkIU1QidjPwiAYu6GcHjg',  # C-SPAN
            'UCupvZG-5ko_eiXAupbDfxWw',  # CNN
            'UCeY0bbntWzzVIaj2z3QigXg',  # NBC News
            'UCBi2mrWuNuyYy4gbM6fU18Q',  # Fox News
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
                if resp.status_code != 200:
                    logger.debug(f"YouTube API returned {resp.status_code} for channel {channel_id}")
                    continue

                data = resp.json()
                if 'error' in data:
                    logger.warning(f"YouTube API error: {data['error'].get('message', 'unknown')}")
                    break  # Likely quota issue, stop all channels

                for item in data.get('items', []):
                    try:
                        video_id = item['id']['videoId']
                        snippet = item['snippet']
                        title = snippet['title']
                        date_str = snippet['publishedAt']
                        date = datetime.fromisoformat(
                            date_str.replace('Z', '+00:00')
                        )

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

    # ==================================================================
    # SOURCE 7: American Presidency Project (UCSB)
    # ==================================================================

    def scrape_presidency_project(self) -> int:
        """Scrape from the American Presidency Project at UCSB.

        The gold standard for official presidential documents including
        speeches, press conferences, and official statements.
        Uses person2=200301 (Trump's ID) for filtering.

        The results page uses a table with columns:
          Date | Related | Document Title
        Title links are in the 3rd <td> cell.
        """
        count = 0
        base_url = 'https://www.presidency.ucsb.edu/advanced-search'

        for page in range(0, 50):
            try:
                params = {
                    'field-keywords': '',
                    'field-keywords2': '',
                    'field-keywords3': '',
                    'from[date]': '',
                    'to[date]': '',
                    'person2': '200301',
                    'page': page,
                }

                resp = self.session.get(base_url, params=params, timeout=60)
                if resp.status_code != 200:
                    logger.debug(f"UCSB page {page} returned {resp.status_code}")
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')

                # Results are in table rows (skip header row)
                rows = soup.select('.view-content table tr')
                data_rows = [r for r in rows if r.find('td')]

                if not data_rows:
                    logger.debug(f"UCSB page {page}: no results, stopping")
                    break

                for row in data_rows:
                    try:
                        cells = row.find_all('td')
                        if len(cells) < 3:
                            continue

                        # Column 0: Date, Column 1: Related, Column 2: Title
                        date_text = cells[0].get_text(strip=True)
                        title_cell = cells[2]
                        title_link = title_cell.select_one('a[href]')

                        if not title_link:
                            continue

                        title = title_link.get_text(strip=True)
                        href = title_link.get('href', '')
                        if not href or not title:
                            continue

                        full_url = (
                            href if href.startswith('http')
                            else f'https://www.presidency.ucsb.edu{href}'
                        )
                        source_id = href.rstrip('/').split('/')[-1]

                        # Parse date from the first column
                        date = datetime.now()
                        if date_text:
                            try:
                                from dateutil import parser as dateparser
                                date = dateparser.parse(date_text)
                            except (ValueError, TypeError):
                                pass

                        # Fetch full transcript from detail page
                        transcript = None
                        try:
                            detail_resp = self.session.get(
                                full_url, timeout=60,
                            )
                            if detail_resp.status_code == 200:
                                detail_soup = BeautifulSoup(
                                    detail_resp.text, 'html.parser',
                                )
                                content = detail_soup.select_one(
                                    '.field-docs-content',
                                )
                                if content:
                                    transcript = content.get_text(
                                        separator=' ', strip=True,
                                    )
                        except Exception as e:
                            logger.debug(
                                f"Error fetching UCSB detail page: {e}",
                            )

                        if self._save_speech(
                            source='presidency_project',
                            source_id=source_id,
                            title=title,
                            date=date,
                            transcript=transcript,
                            source_url=full_url,
                        ):
                            count += 1
                            logger.debug(f"UCSB: saved '{title[:60]}'")

                        time.sleep(1)  # Be polite to UCSB servers

                    except Exception as e:
                        logger.debug(f"Error parsing UCSB row: {e}")

                time.sleep(2)

            except Exception as e:
                logger.warning(f"Error on UCSB page {page}: {e}")
                break

        return count

    # ==================================================================
    # SOURCE 8: Trump Twitter / Truth Social Archive
    # ==================================================================

    def scrape_trump_twitter_archive(self) -> int:
        """Scrape from The Trump Archive (historical tweets/posts).

        NOTE: thetrumparchive.com is now a JS-only shell — all data is
        loaded client-side with no server-rendered content.  Tweet data
        is covered by the SocialMediaImporter bulk import (~56K tweets)
        which runs automatically in pipeline Phase 0.
        Keeping as a stub to avoid breaking scrape_all_sources().
        """
        logger.debug("Trump Twitter Archive is JS-only; tweets covered by SocialMediaImporter")
        return 0

    # ==================================================================
    # SOURCE 9: C-SPAN Video Library (with transcripts)
    # ==================================================================

    def scrape_cspan_transcripts(self) -> int:
        """Scrape C-SPAN video library with interactive transcripts.

        NOTE: C-SPAN search is fully JS-rendered — no video links appear
        in raw HTML.  C-SPAN content is picked up by Google News RSS
        (source #2) and YouTube yt-dlp (source #8).
        Keeping as a stub to avoid breaking scrape_all_sources().
        """
        logger.debug("C-SPAN transcripts search is JS-rendered; coverage via Google News RSS + yt-dlp")
        return 0

    # ==================================================================
    # SOURCE 10: YouTube via yt-dlp (auto-subtitles)
    # ==================================================================

    def scrape_youtube_yt_dlp(self) -> int:
        """Scrape YouTube transcripts from the White House channel and
        Trump-related channels using yt-dlp for video listing and
        youtube-transcript-api for transcript extraction.

        Channels scraped:
          - @WhiteHouse — official White House uploads
          - @RSBNetwork — Right Side Broadcasting (rallies, events)

        No API key required.  yt-dlp lists videos from each channel's
        /videos tab; youtube-transcript-api pulls the transcript for
        each video.  Falls back to yt-dlp subtitle extraction when the
        transcript API fails.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.warning(
                "yt-dlp not installed (pip install yt-dlp), skipping",
            )
            return 0

        count = 0

        # Channel URLs → scrape the /videos tab for each
        channels = [
            ('https://www.youtube.com/@WhiteHouse/videos', 100),
            ('https://www.youtube.com/@RSBNetwork/videos', 50),
        ]

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,        # fast: just list video IDs
            'skip_download': True,
        }

        video_ids: list[tuple[str, str, str]] = []  # (id, title, channel_url)

        # Phase 1: list video IDs from each channel
        for channel_url, limit in channels:
            try:
                opts = {**ydl_opts, 'playlistend': limit}
                with yt_dlp.YoutubeDL(opts) as ydl:
                    result = ydl.extract_info(channel_url, download=False)
                    if not result:
                        continue
                    entries = result.get('entries') or []
                    for entry in entries:
                        if not entry:
                            continue
                        vid = entry.get('id', '')
                        title = entry.get('title', '')
                        if vid and title:
                            video_ids.append((vid, title, channel_url))
                logger.info(
                    f"yt-dlp: listed {len(entries)} videos from {channel_url}",
                )
            except Exception as e:
                logger.warning(f"yt-dlp channel listing error for {channel_url}: {e}")

        if not video_ids:
            logger.debug("yt-dlp: no videos found from any channel")
            return 0

        logger.info(f"yt-dlp: processing {len(video_ids)} videos for transcripts")

        # Phase 2: fetch metadata + transcript for each video
        detail_opts = {
            'quiet': True,
            'no_warnings': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'extract_flat': False,
        }

        for video_id, flat_title, channel_url in video_ids:
            try:
                # Get full metadata via yt-dlp
                with yt_dlp.YoutubeDL(detail_opts) as ydl:
                    entry = ydl.extract_info(
                        f'https://www.youtube.com/watch?v={video_id}',
                        download=False,
                    )
                if not entry:
                    continue

                title = entry.get('title', flat_title)
                upload_date = entry.get('upload_date', '')
                duration = entry.get('duration', 0)
                description = entry.get('description', '')
                channel_name = entry.get('channel', '')
                view_count = entry.get('view_count')

                # Skip very short clips (< 1 min)
                if duration and duration < 60:
                    continue

                # Parse upload date (YYYYMMDD)
                date = datetime.now()
                if upload_date and len(upload_date) == 8:
                    try:
                        date = datetime.strptime(upload_date, '%Y%m%d')
                    except ValueError:
                        pass

                # Transcript: try youtube-transcript-api first (cleanest)
                transcript = self._get_youtube_transcript(video_id)

                # Fallback: yt-dlp subtitle extraction
                if not transcript:
                    try:
                        subs = (
                            entry.get('automatic_captions', {}).get('en', [])
                        )
                        if not subs:
                            subs = entry.get('subtitles', {}).get('en', [])

                        for sub in subs:
                            ext = sub.get('ext', '')
                            if ext in ('json3', 'srv3', 'vtt', 'srt'):
                                sub_resp = self.session.get(
                                    sub['url'], timeout=30,
                                )
                                if sub_resp.status_code == 200:
                                    transcript = self._parse_subtitle_text(
                                        sub_resp.text, ext,
                                    )
                                    if transcript:
                                        break
                    except Exception as e:
                        logger.debug(
                            f"yt-dlp subtitle extraction error for "
                            f"{video_id}: {e}",
                        )

                video_url = f'https://www.youtube.com/watch?v={video_id}'

                if self._save_speech(
                    source='youtube_yt_dlp',
                    source_id=video_id,
                    title=title,
                    date=date,
                    transcript=transcript,
                    source_url=video_url,
                    duration=duration,
                    metadata={
                        'channel': channel_name,
                        'view_count': view_count,
                        'description': (description or '')[:500],
                    },
                ):
                    count += 1
                    word_count = len(transcript.split()) if transcript else 0
                    logger.debug(
                        f"yt-dlp: saved '{title[:50]}' "
                        f"({word_count} words, {duration or '?'}s)",
                    )

            except Exception as e:
                logger.debug(f"Error processing video {video_id}: {e}")

        return count

    def _parse_subtitle_text(self, raw_text: str,
                              fmt: str) -> Optional[str]:
        """Parse subtitle file content (json3 / vtt / srt) to plain text."""
        if fmt == 'json3':
            try:
                data = json.loads(raw_text)
                events = data.get('events', [])
                texts = []
                for event in events:
                    for seg in event.get('segs', []):
                        t = seg.get('utf8', '').strip()
                        if t and t != '\n':
                            texts.append(t)
                return ' '.join(texts) if texts else None
            except (json.JSONDecodeError, ValueError):
                return None

        # VTT / SRT: strip timestamps and metadata lines
        lines = raw_text.split('\n')
        text_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(('WEBVTT', 'NOTE', 'Kind:', 'Language:')):
                continue
            if re.match(r'^\d+$', line):            # SRT sequence number
                continue
            if re.match(r'[\d:.,]+ --> [\d:.,]+', line):   # timestamp
                continue
            line = re.sub(r'<[^>]+>', '', line)             # strip HTML tags
            if line:
                text_lines.append(line)

        return ' '.join(text_lines) if text_lines else None
