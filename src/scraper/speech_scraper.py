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
        """Save a speech to database if not already present. Returns True if new."""
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
        """Attempt to get a YouTube video transcript."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join(item['text'] for item in transcript_list)
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
        """Scrape remarks from whitehouse.gov briefing room.

        Official speeches and remarks:
          - Main page: /briefing-room/speeches-remarks/
          - Pagination: /page/N/
          - Content in <section class="body-content"> or <div class="entry-content">
        """
        count = 0
        base_url = 'https://www.whitehouse.gov/briefing-room/speeches-remarks/'

        for page in range(1, 50):
            try:
                if page == 1:
                    url = base_url
                else:
                    url = f'{base_url}page/{page}/'

                resp = self.session.get(url, timeout=30)
                if resp.status_code != 200:
                    logger.debug(f"WH page {page} returned {resp.status_code}")
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')

                # Find all internal links to /remarks/ or /videos/ paths
                links = soup.find_all('a', href=True)
                page_links = []
                for link in links:
                    href = link['href']
                    # Match remark detail pages like /remarks/2025/06/...
                    # or video pages like /videos/...
                    if re.search(r'/(remarks|speeches-remarks|briefing-room)/\d{4}/', href) or re.search(r'/videos/', href):
                        full_url = href if href.startswith('http') else f'https://www.whitehouse.gov{href}'
                        title = link.get_text(strip=True)
                        if title and len(title) > 5:
                            page_links.append((title, full_url))

                # Also try card-style elements
                for card in soup.select('.wp-block-post, .briefing-statement, article, .news-item'):
                    link_tag = card.select_one('a[href]')
                    if link_tag:
                        href = link_tag['href']
                        if '/remarks/' in href or '/videos/' in href:
                            full_url = href if href.startswith('http') else f'https://www.whitehouse.gov{href}'
                            title = link_tag.get_text(strip=True)
                            if title and len(title) > 5:
                                page_links.append((title, full_url))

                # Deduplicate
                seen = set()
                unique_links = []
                for title, url in page_links:
                    if url not in seen:
                        seen.add(url)
                        unique_links.append((title, url))

                if not unique_links:
                    logger.debug(f"WH page {page}: no remark links found")
                    break

                for title, href in unique_links:
                    try:
                        source_id = href.rstrip('/').split('/')[-1]
                        if not source_id:
                            source_id = '-'.join(href.rstrip('/').split('/')[-3:])

                        date = self._extract_date_from_text(title)
                        # Also try extracting from URL  /remarks/2025/06/12/...
                        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                        if date_match:
                            try:
                                date = datetime(
                                    int(date_match.group(1)),
                                    int(date_match.group(2)),
                                    int(date_match.group(3)),
                                )
                            except ValueError:
                                pass

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

        Factba.se was acquired by Roll Call and now redirects to
        rollcall.com/factbase/. We scrape that plus search for transcripts.
        """
        count = 0

        urls_to_try = [
            'https://rollcall.com/factbase/',
            'https://rollcall.com/section/factbase/',
            'https://rollcall.com/?s=trump+transcript',
        ]

        for base_url in urls_to_try:
            try:
                resp = self.session.get(base_url, timeout=30)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, 'html.parser')

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    title = link.get_text(strip=True)

                    if not title or len(title) < 10:
                        continue

                    # Look for transcript-related articles
                    combined = (title + ' ' + href).lower()
                    if not any(kw in combined for kw in [
                        'transcript', 'trump', 'remarks', 'speech', 'address',
                    ]):
                        continue

                    full_url = href if href.startswith('http') else urljoin(base_url, href)
                    source_id = urlparse(full_url).path.rstrip('/').split('/')[-1]

                    if not source_id or len(source_id) < 3:
                        continue

                    date = self._extract_date_from_text(title)
                    transcript = self._fetch_article_text(full_url)

                    if self._save_speech(
                        source='rollcall_factbase',
                        source_id=source_id,
                        title=title,
                        date=date,
                        transcript=transcript,
                        source_url=full_url,
                    ):
                        count += 1
                        logger.debug(f"RollCall: saved '{title[:60]}'")

            except Exception as e:
                logger.warning(f"Roll Call scraper error for {base_url}: {e}")

        return count

    # ==================================================================
    # SOURCE 5: C-SPAN
    # ==================================================================

    def scrape_cspan(self) -> int:
        """Scrape Trump appearances from C-SPAN search."""
        count = 0

        try:
            search_url = 'https://www.c-span.org/search/'
            params = {
                'query': 'trump',
                'sort': 'Most+Recent',
                'type': 'videos',
            }

            resp = self.session.get(search_url, params=params, timeout=30)
            if resp.status_code != 200:
                logger.debug(f"C-SPAN returned {resp.status_code}")
                return 0

            soup = BeautifulSoup(resp.text, 'html.parser')

            # C-SPAN uses /video/ links for their content
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/video/' not in href:
                    continue

                title = link.get_text(strip=True)
                if not title or len(title) < 10:
                    continue

                full_url = href if href.startswith('http') else f'https://www.c-span.org{href}'
                source_id = href.rstrip('/').split('/')[-1]
                # Remove query params from source_id
                source_id = source_id.split('?')[0]

                if not source_id:
                    continue

                date = self._extract_date_from_text(
                    link.parent.get_text(' ', strip=True) if link.parent else title
                )

                if self._save_speech(
                    source='cspan',
                    source_id=source_id,
                    title=title,
                    date=date,
                    source_url=full_url,
                    speech_type='cspan_appearance',
                ):
                    count += 1

        except Exception as e:
            logger.warning(f"C-SPAN scraper error: {e}")

        return count

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

                resp = self.session.get(base_url, params=params, timeout=30)
                if resp.status_code != 200:
                    logger.debug(f"UCSB page {page} returned {resp.status_code}")
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')
                rows = soup.select('.views-row')

                if not rows:
                    logger.debug(f"UCSB page {page}: no results, stopping")
                    break

                for row in rows:
                    try:
                        title_el = row.select_one('.field-title a, h3 a, a')
                        if not title_el:
                            continue

                        title = title_el.get_text(strip=True)
                        href = title_el.get('href', '')
                        if not href:
                            continue

                        full_url = (
                            href if href.startswith('http')
                            else f'https://www.presidency.ucsb.edu{href}'
                        )
                        source_id = href.rstrip('/').split('/')[-1]

                        # Extract date
                        date_el = row.select_one('.date-display-single')
                        date = datetime.now()
                        if date_el:
                            try:
                                from dateutil import parser as dateparser
                                date = dateparser.parse(
                                    date_el.get_text(strip=True),
                                )
                            except (ValueError, TypeError):
                                pass

                        # Fetch full transcript from detail page
                        transcript = None
                        try:
                            detail_resp = self.session.get(
                                full_url, timeout=30,
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

        The archive typically serves its data as JSON—either via a
        direct API endpoint or embedded in the page source.  Useful for
        short-form language patterns and vocabulary mapping.
        """
        count = 0

        endpoints = [
            'https://www.thetrumparchive.com/latest-tweets',
            'https://www.thetrumparchive.com/',
        ]

        for endpoint in endpoints:
            try:
                resp = self.session.get(endpoint, timeout=30)
                if resp.status_code != 200:
                    continue

                # Attempt 1: response is raw JSON
                try:
                    data = resp.json()
                    tweets = (
                        data if isinstance(data, list)
                        else data.get('tweets', data.get('data', []))
                    )

                    for tweet in tweets:
                        text = tweet.get('text', tweet.get('content', ''))
                        if not text or len(text) < 20:
                            continue

                        tweet_id = str(
                            tweet.get('id', tweet.get('tweetId', '')),
                        )
                        date_str = tweet.get(
                            'date', tweet.get('created_at', ''),
                        )
                        date = datetime.now()
                        if date_str:
                            try:
                                from dateutil import parser as dateparser
                                date = dateparser.parse(date_str)
                            except (ValueError, TypeError):
                                pass

                        short_title = (
                            f'Tweet: {text[:80]}...'
                            if len(text) > 80
                            else f'Tweet: {text}'
                        )
                        if self._save_speech(
                            source='twitter_archive',
                            source_id=(
                                tweet_id
                                or f'tweet_{hash(text) % 10**8}'
                            ),
                            title=short_title,
                            date=date,
                            transcript=text,
                            speech_type='social_media',
                        ):
                            count += 1

                    if count > 0:
                        logger.info(
                            f"Twitter Archive: loaded {count} posts "
                            f"from JSON endpoint",
                        )
                        return count

                except (json.JSONDecodeError, ValueError):
                    pass

                # Attempt 2: JSON data embedded in HTML <script> tags
                soup = BeautifulSoup(resp.text, 'html.parser')
                for script in soup.find_all('script'):
                    script_text = script.string or ''
                    if 'tweet' not in script_text.lower() or '{' not in script_text:
                        continue

                    json_match = re.search(
                        r'(\[.*?\])', script_text, re.DOTALL,
                    )
                    if not json_match:
                        continue

                    try:
                        tweets = json.loads(json_match.group(1))
                        for tweet in tweets[:5000]:
                            tweet_text = tweet.get('text', '')
                            if tweet_text and len(tweet_text) > 20:
                                tweet_id = str(tweet.get('id', ''))
                                if self._save_speech(
                                    source='twitter_archive',
                                    source_id=(
                                        tweet_id
                                        or f'tweet_{hash(tweet_text) % 10**8}'
                                    ),
                                    title=f'Tweet: {tweet_text[:80]}',
                                    date=datetime.now(),
                                    transcript=tweet_text,
                                    speech_type='social_media',
                                ):
                                    count += 1
                    except (json.JSONDecodeError, ValueError):
                        continue

            except Exception as e:
                logger.debug(f"Twitter Archive error for {endpoint}: {e}")

        return count

    # ==================================================================
    # SOURCE 9: C-SPAN Video Library (with transcripts)
    # ==================================================================

    def scrape_cspan_transcripts(self) -> int:
        """Scrape C-SPAN video library with interactive transcripts.

        Uses Trump's person ID (92774) to search the C-SPAN archive.
        Attempts to extract transcript data from video detail pages
        where it is often embedded as JSON or rendered in a transcript div.
        """
        count = 0
        base_url = 'https://www.c-span.org/search/'

        for page in range(1, 20):
            try:
                params = {
                    'sdate': '',
                    'edate': '',
                    'searchtype': 'Videos',
                    'sort': 'Most+Recent',
                    'text': '0',
                    'personid[]': '92774',
                    'page': page,
                }

                resp = self.session.get(
                    base_url, params=params, timeout=30,
                )
                if resp.status_code != 200:
                    logger.debug(
                        f"C-SPAN search page {page} returned "
                        f"{resp.status_code}",
                    )
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')
                video_items = soup.select(
                    '.video-result, .search-result, li',
                )

                found_any = False
                for item in video_items:
                    try:
                        link = item.select_one('a[href*="/video/"]')
                        if not link:
                            continue

                        found_any = True
                        href = link['href']
                        title = link.get_text(strip=True)
                        if not title or len(title) < 5:
                            continue

                        full_url = (
                            href if href.startswith('http')
                            else f'https://www.c-span.org{href}'
                        )
                        source_id = re.sub(
                            r'[?#].*', '',
                            href.rstrip('/').split('/')[-1],
                        )
                        if not source_id or len(source_id) < 2:
                            continue

                        # Date extraction
                        date_el = item.select_one('.date, time, .air-date')
                        date = datetime.now()
                        if date_el:
                            try:
                                from dateutil import parser as dateparser
                                date_text = (
                                    date_el.get('datetime')
                                    or date_el.get_text(strip=True)
                                )
                                date = dateparser.parse(date_text)
                            except (ValueError, TypeError):
                                pass

                        # Fetch detail page and extract transcript
                        transcript = None
                        try:
                            detail_resp = self.session.get(
                                full_url, timeout=30,
                            )
                            if detail_resp.status_code == 200:
                                detail_text = detail_resp.text

                                # Try JSON transcript embedded in page
                                t_match = re.search(
                                    r'transcript["\']?\s*:\s*(\[.*?\])',
                                    detail_text,
                                    re.DOTALL,
                                )
                                if t_match:
                                    try:
                                        segments = json.loads(
                                            t_match.group(1),
                                        )
                                        transcript = ' '.join(
                                            s.get('text', '')
                                            for s in segments
                                            if isinstance(s, dict)
                                        )
                                    except (json.JSONDecodeError, ValueError):
                                        pass

                                # Fallback: HTML transcript div
                                if not transcript:
                                    detail_soup = BeautifulSoup(
                                        detail_text, 'html.parser',
                                    )
                                    t_div = detail_soup.select_one(
                                        '.transcript-text, '
                                        '#annotator-source-text, '
                                        '.text-body',
                                    )
                                    if t_div:
                                        transcript = t_div.get_text(
                                            separator=' ', strip=True,
                                        )
                        except Exception as e:
                            logger.debug(
                                f"Error fetching C-SPAN detail: {e}",
                            )

                        if self._save_speech(
                            source='cspan_transcripts',
                            source_id=source_id,
                            title=title,
                            date=date,
                            transcript=transcript,
                            source_url=full_url,
                            speech_type='cspan_appearance',
                        ):
                            count += 1
                            logger.debug(
                                f"C-SPAN transcript: saved '{title[:60]}'",
                            )

                        time.sleep(1.5)

                    except Exception as e:
                        logger.debug(f"Error parsing C-SPAN result: {e}")

                if not found_any:
                    break

                time.sleep(2)

            except Exception as e:
                logger.warning(
                    f"C-SPAN transcript search error on page {page}: {e}",
                )
                break

        return count

    # ==================================================================
    # SOURCE 10: YouTube via yt-dlp (auto-subtitles)
    # ==================================================================

    def scrape_youtube_yt_dlp(self) -> int:
        """Scrape YouTube transcripts using yt-dlp auto-generated subtitles.

        Downloads auto-subtitles from RSBN, C-SPAN, White House, and
        news channels.  Falls back to youtube-transcript-api when
        yt-dlp is unavailable.  More reliable than the YouTube Data API
        for transcript extraction and does not require an API key.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.warning(
                "yt-dlp not installed (pip install yt-dlp), skipping",
            )
            return 0

        count = 0

        search_queries = [
            'ytsearch20:trump full speech 2026',
            'ytsearch20:trump remarks 2026',
            'ytsearch20:trump press conference 2026',
            'ytsearch10:trump rally full 2025',
        ]

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'extract_flat': False,
        }

        for query in search_queries:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    results = ydl.extract_info(query, download=False)
                    if not results or 'entries' not in results:
                        continue

                    for entry in (results.get('entries') or []):
                        if not entry:
                            continue

                        try:
                            video_id = entry.get('id', '')
                            title = entry.get('title', '')
                            upload_date = entry.get('upload_date', '')
                            duration = entry.get('duration', 0)

                            if not video_id or not title:
                                continue

                            # Skip short clips (< 5 min)
                            if duration and duration < 300:
                                continue

                            # Parse upload date (YYYYMMDD)
                            date = datetime.now()
                            if upload_date and len(upload_date) == 8:
                                try:
                                    date = datetime.strptime(
                                        upload_date, '%Y%m%d',
                                    )
                                except ValueError:
                                    pass

                            # Try youtube-transcript-api first
                            transcript = self._get_youtube_transcript(
                                video_id,
                            )

                            # Fall back to yt-dlp subtitle extraction
                            if not transcript:
                                try:
                                    subs = (
                                        entry.get(
                                            'automatic_captions', {},
                                        ).get('en', [])
                                    )
                                    if not subs:
                                        subs = (
                                            entry.get(
                                                'subtitles', {},
                                            ).get('en', [])
                                        )

                                    for sub in subs:
                                        ext = sub.get('ext', '')
                                        if ext in (
                                            'json3', 'srv3', 'vtt', 'srt',
                                        ):
                                            sub_resp = self.session.get(
                                                sub['url'], timeout=30,
                                            )
                                            if sub_resp.status_code == 200:
                                                transcript = (
                                                    self._parse_subtitle_text(
                                                        sub_resp.text, ext,
                                                    )
                                                )
                                                if transcript:
                                                    break
                                except Exception as e:
                                    logger.debug(
                                        f"yt-dlp subtitle extraction "
                                        f"error: {e}",
                                    )

                            if self._save_speech(
                                source='youtube_yt_dlp',
                                source_id=video_id,
                                title=title,
                                date=date,
                                transcript=transcript,
                                source_url=(
                                    'https://www.youtube.com'
                                    f'/watch?v={video_id}'
                                ),
                                duration=duration,
                            ):
                                count += 1
                                logger.debug(
                                    f"yt-dlp: saved '{title[:60]}'",
                                )

                        except Exception as e:
                            logger.debug(
                                f"Error processing yt-dlp entry: {e}",
                            )

            except Exception as e:
                logger.debug(f"yt-dlp search error for '{query}': {e}")

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
