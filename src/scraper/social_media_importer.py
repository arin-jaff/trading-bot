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
        """Clean social media post text for training."""
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
