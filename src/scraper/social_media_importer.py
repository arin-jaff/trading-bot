"""Bulk importer for Twitter archive and Truth Social posts.

Downloads/parses full archives and saves both individual posts and daily
digest records (grouped by date) so the Markov trainer has enough text
per record to train on.
"""

import csv
import io
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional
from loguru import logger

from ..database.db import get_session
from ..database.models import Speech


# Minimum word count for a daily digest to be useful for Markov training
MIN_DIGEST_WORDS = 50


class SocialMediaImporter:
    """Import Trump tweets and Truth Social posts into the speech corpus."""

    def __init__(self):
        self.imports_dir = os.path.join('data', 'imports')
        os.makedirs(self.imports_dir, exist_ok=True)
        self._status = {
            'state': 'idle',  # idle, downloading, importing, complete, error
            'source': '',
            'progress': 0.0,
            'total_posts': 0,
            'imported': 0,
            'skipped_dupes': 0,
            'daily_digests_created': 0,
            'error': None,
        }

    def get_status(self) -> dict:
        return self._status.copy()

    # ── Twitter Archive ──

    def download_twitter_archive(self) -> str:
        """Download full Trump tweet archive CSV from thetrumparchive.com.

        Returns path to downloaded file.
        """
        import requests

        self._status.update(
            state='downloading', source='twitter',
            progress=0.0, error=None,
        )

        url = 'https://www.thetrumparchive.com/latest-tweets'
        csv_url = 'https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6'
        json_url = 'https://raw.githubusercontent.com/brown-uk/trumptweets/master/tweets.json'

        dest_path = os.path.join(self.imports_dir, 'trump_tweets.json')

        # Try multiple sources
        for attempt_url in [json_url, csv_url]:
            try:
                logger.info(f"Downloading Twitter archive from {attempt_url}")
                r = requests.get(attempt_url, timeout=120, stream=True)
                r.raise_for_status()

                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                logger.info(f"Downloaded {size_mb:.1f}MB to {dest_path}")
                self._status['progress'] = 0.1
                return dest_path

            except Exception as e:
                logger.warning(f"Download from {attempt_url} failed: {e}")
                continue

        self._status.update(state='error', error='All download sources failed')
        raise RuntimeError("Could not download Twitter archive from any source")

    def import_twitter_archive(self, file_path: Optional[str] = None) -> dict:
        """Parse and import Trump Twitter archive.

        Accepts JSON (list of {text, date, ...}) or CSV (columns: id, text, date, ...).
        Saves individual tweets + daily digests.
        """
        if not file_path:
            file_path = os.path.join(self.imports_dir, 'trump_tweets.json')

        if not os.path.exists(file_path):
            # Try to download first
            file_path = self.download_twitter_archive()

        self._status.update(
            state='importing', source='twitter',
            progress=0.1, error=None,
        )

        try:
            posts = self._parse_archive_file(file_path)
            self._status['total_posts'] = len(posts)
            logger.info(f"Parsed {len(posts)} tweets from {file_path}")

            result = self._save_posts_and_digests(posts, 'twitter')
            self._status.update(state='complete', progress=1.0, **result)
            logger.info(f"Twitter import complete: {result}")
            return result

        except Exception as e:
            self._status.update(state='error', error=str(e))
            logger.error(f"Twitter import failed: {e}")
            raise

    def import_truth_social(self, file_path: str) -> dict:
        """Import Truth Social posts from a JSON dump.

        Expected format: list of {text, date, ...} or {content, created_at, ...}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Truth Social dump not found: {file_path}")

        self._status.update(
            state='importing', source='truth_social',
            progress=0.1, error=None,
        )

        try:
            posts = self._parse_archive_file(file_path, source='truth_social')
            self._status['total_posts'] = len(posts)
            logger.info(f"Parsed {len(posts)} Truth Social posts from {file_path}")

            result = self._save_posts_and_digests(posts, 'truth_social')
            self._status.update(state='complete', progress=1.0, **result)
            logger.info(f"Truth Social import complete: {result}")
            return result

        except Exception as e:
            self._status.update(state='error', error=str(e))
            logger.error(f"Truth Social import failed: {e}")
            raise

    # ── Parsing ──

    def _parse_archive_file(self, file_path: str,
                            source: str = 'twitter') -> list[dict]:
        """Parse a JSON or CSV archive file into a list of {text, date} dicts."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read().strip()

        posts = []

        # Try JSON first
        if content.startswith('[') or content.startswith('{'):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # Might be {data: [...]} or similar wrapper
                    for key in ('data', 'tweets', 'posts', 'results'):
                        if key in data and isinstance(data[key], list):
                            data = data[key]
                            break
                    else:
                        data = [data]

                for item in data:
                    text = (item.get('text') or item.get('content')
                            or item.get('full_text') or item.get('body') or '')
                    date_str = (item.get('date') or item.get('created_at')
                                or item.get('timestamp') or '')
                    date = self._parse_date(date_str)
                    if text.strip():
                        posts.append({
                            'text': self._clean_post_text(text),
                            'date': date,
                            'source_id': str(item.get('id', '')),
                        })
                return posts
            except json.JSONDecodeError:
                pass

        # Try CSV
        try:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                text = (row.get('text') or row.get('content')
                        or row.get('full_text') or row.get('body') or '')
                date_str = (row.get('date') or row.get('created_at')
                            or row.get('timestamp') or '')
                date = self._parse_date(date_str)
                if text.strip():
                    posts.append({
                        'text': self._clean_post_text(text),
                        'date': date,
                        'source_id': str(row.get('id', '')),
                    })
            return posts
        except Exception:
            pass

        raise ValueError(f"Could not parse {file_path} as JSON or CSV")

    @staticmethod
    def _clean_post_text(text: str) -> str:
        """Clean social media post text for training.

        Strips HTML tags, URLs, @mentions, and normalizes whitespace.
        """
        # Strip HTML tags (Truth Social / Mastodon returns HTML)
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode common HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove RT prefix
        text = re.sub(r'^RT\s+', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse various date formats from social media archives."""
        if not date_str:
            return datetime(2020, 1, 1)

        if isinstance(date_str, datetime):
            return date_str

        for fmt in (
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%S%z',
            '%a %b %d %H:%M:%S %z %Y',  # Twitter format
            '%m/%d/%Y %H:%M',
            '%Y-%m-%d',
        ):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        # Last resort: try dateutil
        try:
            from dateutil import parser as dateutil_parser
            return dateutil_parser.parse(date_str)
        except Exception:
            return datetime(2020, 1, 1)

    # ── Grouping + Saving ──

    def _group_into_daily_digests(self, posts: list[dict],
                                  source_name: str) -> list[dict]:
        """Group posts by date into daily digest Speech records.

        Each digest concatenates all posts from one day, prefixed with the
        date, so the Markov trainer has enough text (>50 words) per record.
        """
        by_date = defaultdict(list)
        for post in posts:
            day_key = post['date'].strftime('%Y-%m-%d')
            by_date[day_key].append(post)

        digests = []
        for day_key in sorted(by_date.keys()):
            day_posts = by_date[day_key]
            combined_text = '\n'.join(p['text'] for p in day_posts)
            word_count = len(combined_text.split())

            if word_count < MIN_DIGEST_WORDS:
                continue

            digests.append({
                'source': source_name,
                'source_id': f'daily_{source_name}_{day_key}',
                'title': f'Trump {source_name.replace("_", " ").title()} — {day_key} ({len(day_posts)} posts)',
                'date': day_posts[0]['date'],
                'transcript': combined_text,
                'word_count': word_count,
                'speech_type': 'social_media_daily',
                'post_count': len(day_posts),
            })

        return digests

    def _save_posts_and_digests(self, posts: list[dict],
                                source_name: str) -> dict:
        """Save individual posts + daily digests to the database.

        Returns summary dict with counts.
        """
        imported = 0
        skipped = 0
        batch_size = 500

        # Save individual posts in batches
        for i in range(0, len(posts), batch_size):
            batch = posts[i:i + batch_size]
            with get_session() as session:
                for post in batch:
                    sid = post.get('source_id') or str(hash(post['text'][:100]))
                    existing = session.query(Speech).filter_by(
                        source=source_name, source_id=sid,
                    ).first()
                    if existing:
                        skipped += 1
                        continue

                    speech = Speech(
                        source=source_name,
                        source_id=sid,
                        title=post['text'][:200],
                        speech_type='social_media',
                        date=post['date'],
                        transcript=post['text'],
                        transcript_source=source_name,
                        word_count=len(post['text'].split()),
                        is_processed=False,
                    )
                    session.add(speech)
                    imported += 1

            # Update progress
            progress = 0.1 + 0.6 * min(1.0, (i + batch_size) / max(1, len(posts)))
            self._status.update(
                progress=progress, imported=imported, skipped_dupes=skipped,
            )

            if (i + batch_size) % 5000 == 0:
                logger.info(f"Imported {imported} individual posts ({skipped} skipped)...")

        logger.info(f"Individual posts: {imported} imported, {skipped} skipped")

        # Create daily digests
        self._status.update(progress=0.75, state='importing')
        digests = self._group_into_daily_digests(posts, source_name)
        digest_count = 0

        with get_session() as session:
            for digest in digests:
                existing = session.query(Speech).filter_by(
                    source=digest['source'], source_id=digest['source_id'],
                ).first()
                if existing:
                    continue

                speech = Speech(
                    source=digest['source'],
                    source_id=digest['source_id'],
                    title=digest['title'],
                    speech_type=digest['speech_type'],
                    date=digest['date'],
                    transcript=digest['transcript'],
                    transcript_source=digest['source'],
                    word_count=digest['word_count'],
                    is_processed=False,
                )
                session.add(speech)
                digest_count += 1

        self._status['daily_digests_created'] = digest_count
        logger.info(f"Created {digest_count} daily digests from {len(digests)} date groups")

        return {
            'imported': imported,
            'skipped_dupes': skipped,
            'daily_digests_created': digest_count,
            'total_posts': len(posts),
        }

    # ── Live Scraping (periodic, for new posts) ──

    def scrape_latest_posts(self) -> int:
        """Scrape recent Truth Social + Twitter/X posts and rebuild daily digests.

        Called periodically by the scheduler. Fetches new posts from both
        platforms, saves them individually, then rebuilds recent daily digests
        so the Markov trainer picks up fresh social media language.

        Returns count of new posts saved.
        """
        total_new = 0

        # 1. Scrape Truth Social RSS/API
        try:
            new_truth = self._scrape_truth_social_feed()
            total_new += new_truth
            if new_truth:
                logger.info(f"Scraped {new_truth} new Truth Social posts")
        except Exception as e:
            logger.warning(f"Truth Social scrape failed: {e}")

        # 2. Scrape recent Twitter/X posts via public proxies
        try:
            new_twitter = self._scrape_twitter_recent()
            total_new += new_twitter
            if new_twitter:
                logger.info(f"Scraped {new_twitter} new Twitter/X posts")
        except Exception as e:
            logger.warning(f"Twitter/X scrape failed: {e}")

        # 3. Rebuild recent daily digests (last 7 days)
        if total_new > 0:
            try:
                digests = self._rebuild_recent_digests(days=7)
                logger.info(f"Rebuilt {digests} daily digests from recent posts")
            except Exception as e:
                logger.warning(f"Daily digest rebuild failed: {e}")

        return total_new

    def ensure_initial_import(self) -> dict:
        """Auto-run the Twitter bulk import if no tweets exist in the DB.

        Called once at pipeline startup. If the DB already has tweets,
        this is a no-op. Returns import result or skip status.
        """
        with get_session() as session:
            existing = session.query(Speech).filter_by(
                source='twitter', speech_type='social_media',
            ).count()

        if existing > 0:
            return {'status': 'skipped', 'existing_tweets': existing}

        logger.info("No tweets in DB — running initial Twitter archive import...")
        try:
            result = self.import_twitter_archive()
            return {'status': 'imported', **result}
        except Exception as e:
            logger.error(f"Initial Twitter import failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _scrape_truth_social_feed(self) -> int:
        """Fetch recent Truth Social posts via public feeds/APIs.

        Tries multiple sources for Truth Social content:
        1. RSS proxy feeds (third-party aggregators)
        2. Direct Truth Social public API (if available)
        3. Web scraping of public profile page
        """
        import requests

        count = 0
        posts = []

        # Source 1: Try common RSS proxy/aggregator feeds for Truth Social
        rss_urls = [
            'https://truthsocial.com/users/realDonaldTrump/feed',
            'https://truthsocial.com/@realDonaldTrump.rss',
        ]

        for url in rss_urls:
            try:
                resp = requests.get(url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; TrumpGPT/1.0)',
                    'Accept': 'application/rss+xml, application/atom+xml, application/json, text/html',
                })
                if resp.status_code != 200:
                    continue

                # Try parsing as RSS/Atom
                import feedparser
                feed = feedparser.parse(resp.text)
                if feed.entries:
                    for entry in feed.entries:
                        text = entry.get('summary', entry.get('title', ''))
                        # Strip HTML tags
                        text = re.sub(r'<[^>]+>', '', text).strip()
                        if not text or len(text) < 10:
                            continue
                        date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                date = datetime(*entry.published_parsed[:6])
                            except Exception:
                                pass
                        posts.append({
                            'text': self._clean_post_text(text),
                            'date': date,
                            'source_id': entry.get('id', entry.get('link', '')),
                        })
                    break

                # Try parsing as JSON
                try:
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get('statuses', data.get('data', []))
                    for item in items:
                        text = item.get('content', item.get('text', ''))
                        text = re.sub(r'<[^>]+>', '', text).strip()
                        if not text or len(text) < 10:
                            continue
                        date_str = item.get('created_at', '')
                        date = self._parse_date(date_str) if date_str else datetime.now()
                        posts.append({
                            'text': self._clean_post_text(text),
                            'date': date,
                            'source_id': str(item.get('id', '')),
                        })
                    if posts:
                        break
                except (json.JSONDecodeError, ValueError):
                    pass

            except Exception as e:
                logger.debug(f"Truth Social feed {url} failed: {e}")
                continue

        # Source 2: Try Mastodon-compatible API (Truth Social is a Mastodon fork)
        if not posts:
            try:
                api_url = 'https://truthsocial.com/api/v1/accounts/107780257626128497/statuses'
                resp = requests.get(api_url, timeout=15, params={'limit': 40}, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; TrumpGPT/1.0)',
                })
                if resp.status_code == 200:
                    for item in resp.json():
                        text = item.get('content', '')
                        text = re.sub(r'<[^>]+>', '', text).strip()
                        if not text or len(text) < 10:
                            continue
                        date = self._parse_date(item.get('created_at', ''))
                        posts.append({
                            'text': self._clean_post_text(text),
                            'date': date,
                            'source_id': str(item.get('id', '')),
                        })
            except Exception as e:
                logger.debug(f"Truth Social Mastodon API failed: {e}")

        # Save new posts
        if posts:
            with get_session() as session:
                for post in posts:
                    sid = post.get('source_id') or str(hash(post['text'][:100]))
                    existing = session.query(Speech).filter_by(
                        source='truth_social', source_id=sid,
                    ).first()
                    if existing:
                        continue

                    speech = Speech(
                        source='truth_social',
                        source_id=sid,
                        title=post['text'][:200],
                        speech_type='social_media',
                        date=post['date'],
                        transcript=post['text'],
                        transcript_source='truth_social_feed',
                        word_count=len(post['text'].split()),
                        is_processed=False,
                    )
                    session.add(speech)
                    count += 1

        return count

    def _scrape_twitter_recent(self) -> int:
        """Scrape recent Trump tweets/posts from X via public proxy services.

        Tries multiple Nitter instances and RSS bridges to get recent tweets
        without requiring Twitter API credentials.
        """
        import requests

        count = 0
        posts = []

        # Nitter instances (public Twitter frontends that serve RSS)
        nitter_instances = [
            'https://nitter.privacydev.net',
            'https://nitter.poast.org',
            'https://nitter.net',
            'https://nitter.cz',
            'https://nitter.1d4.us',
        ]

        for instance in nitter_instances:
            try:
                rss_url = f'{instance}/realDonaldTrump/rss'
                resp = requests.get(rss_url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
                })
                if resp.status_code != 200:
                    continue

                import feedparser
                feed = feedparser.parse(resp.text)
                if not feed.entries:
                    continue

                for entry in feed.entries:
                    text = entry.get('title', entry.get('summary', ''))
                    # Strip HTML (Nitter includes some)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = self._clean_post_text(text)
                    if not text or len(text) < 10:
                        continue

                    date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            date = datetime(*entry.published_parsed[:6])
                        except Exception:
                            pass

                    posts.append({
                        'text': text,
                        'date': date,
                        'source_id': entry.get('id', entry.get('link', '')),
                    })

                logger.info(f"Twitter/X: got {len(posts)} posts from {instance}")
                break  # Got data, stop trying other instances

            except Exception as e:
                logger.debug(f"Nitter instance {instance} failed: {e}")
                continue

        # Fallback: Try RSSBridge
        if not posts:
            rssbridge_urls = [
                'https://rss-bridge.org/bridge01/?action=display&bridge=TwitterBridge&context=By+username&u=realDonaldTrump&norep=on&noretweet=on&format=Atom',
            ]
            for url in rssbridge_urls:
                try:
                    resp = requests.get(url, timeout=15)
                    if resp.status_code != 200:
                        continue
                    import feedparser
                    feed = feedparser.parse(resp.text)
                    for entry in feed.entries:
                        text = entry.get('summary', entry.get('title', ''))
                        text = re.sub(r'<[^>]+>', ' ', text)
                        text = self._clean_post_text(text)
                        if not text or len(text) < 10:
                            continue
                        date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                date = datetime(*entry.published_parsed[:6])
                            except Exception:
                                pass
                        posts.append({
                            'text': text,
                            'date': date,
                            'source_id': entry.get('id', entry.get('link', '')),
                        })
                    if posts:
                        logger.info(f"Twitter/X: got {len(posts)} posts from RSS Bridge")
                        break
                except Exception as e:
                    logger.debug(f"RSS Bridge failed: {e}")

        # Save new posts
        if posts:
            with get_session() as session:
                for post in posts:
                    sid = post.get('source_id') or f'tweet_{hash(post["text"][:100]) % 10**8}'
                    existing = session.query(Speech).filter_by(
                        source='twitter', source_id=sid,
                    ).first()
                    if existing:
                        continue

                    speech = Speech(
                        source='twitter',
                        source_id=sid,
                        title=post['text'][:200],
                        speech_type='social_media',
                        date=post['date'],
                        transcript=post['text'],
                        transcript_source='twitter_live',
                        word_count=len(post['text'].split()),
                        is_processed=False,
                    )
                    session.add(speech)
                    count += 1

        return count

    def clean_existing_posts(self) -> int:
        """Strip HTML from all existing social media posts in the DB.

        One-time cleanup for posts that were imported before the HTML
        stripping was added to _clean_post_text(). Safe to run multiple
        times — already-clean posts won't be modified.
        """
        cleaned = 0
        html_pattern = re.compile(r'<[^>]+>')

        with get_session() as session:
            dirty_posts = session.query(Speech).filter(
                Speech.speech_type == 'social_media',
                Speech.transcript.isnot(None),
                Speech.transcript.contains('<'),  # quick filter
            ).all()

            for post in dirty_posts:
                if not html_pattern.search(post.transcript):
                    continue

                original = post.transcript
                cleaned_text = self._clean_post_text(original)

                if cleaned_text != original and cleaned_text:
                    post.transcript = cleaned_text
                    post.title = cleaned_text[:200]
                    post.word_count = len(cleaned_text.split())
                    post.is_processed = False  # re-process for term analysis
                    cleaned += 1

            # Also clean daily digests
            dirty_digests = session.query(Speech).filter(
                Speech.speech_type == 'social_media_daily',
                Speech.transcript.isnot(None),
                Speech.transcript.contains('<'),
            ).all()

            for digest in dirty_digests:
                if not html_pattern.search(digest.transcript):
                    continue
                # Clean each line separately (digests are newline-joined posts)
                lines = digest.transcript.split('\n')
                cleaned_lines = [self._clean_post_text(line) for line in lines]
                cleaned_lines = [l for l in cleaned_lines if l]
                new_text = '\n'.join(cleaned_lines)
                if new_text != digest.transcript:
                    digest.transcript = new_text
                    digest.word_count = len(new_text.split())
                    digest.is_processed = False
                    cleaned += 1

        if cleaned:
            logger.info(f"Cleaned HTML from {cleaned} social media posts/digests")
        return cleaned

    def _rebuild_recent_digests(self, days: int = 7) -> int:
        """Rebuild daily digests for the last N days from individual posts.

        This handles the case where new posts have been scraped since the
        last digest was created. Existing digests are updated in-place
        (transcript replaced with full day's content).
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)
        digest_count = 0

        with get_session() as session:
            # Get all recent individual social media posts
            recent_posts = session.query(Speech).filter(
                Speech.speech_type == 'social_media',
                Speech.date >= cutoff,
                Speech.transcript.isnot(None),
            ).order_by(Speech.date).all()

            if not recent_posts:
                return 0

            # Group by source + date
            by_key = defaultdict(list)
            for post in recent_posts:
                source = post.source
                day_key = post.date.strftime('%Y-%m-%d')
                by_key[(source, day_key)].append(post.transcript)

            # Create or update daily digests
            for (source, day_key), texts in by_key.items():
                combined = '\n'.join(texts)
                word_count = len(combined.split())
                if word_count < MIN_DIGEST_WORDS:
                    continue

                digest_sid = f'daily_{source}_{day_key}'
                existing = session.query(Speech).filter_by(
                    source=source, source_id=digest_sid,
                ).first()

                if existing:
                    # Update existing digest if content has grown
                    if word_count > (existing.word_count or 0):
                        existing.transcript = combined
                        existing.word_count = word_count
                        existing.title = f'Trump {source.replace("_", " ").title()} — {day_key} ({len(texts)} posts)'
                        existing.is_processed = False  # re-process for term analysis
                else:
                    speech = Speech(
                        source=source,
                        source_id=digest_sid,
                        title=f'Trump {source.replace("_", " ").title()} — {day_key} ({len(texts)} posts)',
                        speech_type='social_media_daily',
                        date=datetime.strptime(day_key, '%Y-%m-%d'),
                        transcript=combined,
                        transcript_source=source,
                        word_count=word_count,
                        is_processed=False,
                    )
                    session.add(speech)
                    digest_count += 1

        return digest_count

    def get_stats(self) -> dict:
        """Get social media corpus stats from the database."""
        with get_session() as session:
            tweets = session.query(Speech).filter_by(
                source='twitter', speech_type='social_media',
            ).count()
            truth_social = session.query(Speech).filter_by(
                source='truth_social', speech_type='social_media',
            ).count()
            daily_digests = session.query(Speech).filter_by(
                speech_type='social_media_daily',
            ).count()
            digest_words = session.query(
                Speech.word_count
            ).filter_by(
                speech_type='social_media_daily',
            ).all()
            total_digest_words = sum(wc for (wc,) in digest_words if wc)

        return {
            'tweets': tweets,
            'truth_social': truth_social,
            'daily_digests': daily_digests,
            'total_digest_words': total_digest_words,
        }
