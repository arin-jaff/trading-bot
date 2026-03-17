"""One-time script to strip HTML from all social media posts in the DB.

Usage:
    python scripts/clean_html_posts.py

Safe to run multiple times — already-clean posts are skipped.
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db import init_db, get_session
from src.database.models import Speech


def clean_post_text(text: str) -> str:
    """Strip HTML tags, decode entities, remove URLs and @mentions."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'^RT\s+', '', text)
    text = ' '.join(text.split())
    return text.strip()


def main():
    init_db()
    html_pattern = re.compile(r'<[^>]+>')
    cleaned = 0
    skipped = 0

    with get_session() as session:
        posts = session.query(Speech).filter(
            Speech.speech_type.in_(['social_media', 'social_media_daily']),
            Speech.transcript.isnot(None),
            Speech.transcript.contains('<'),
        ).all()

        print(f"Found {len(posts)} posts with potential HTML")

        for post in posts:
            if not html_pattern.search(post.transcript):
                skipped += 1
                continue

            if post.speech_type == 'social_media_daily':
                lines = post.transcript.split('\n')
                cleaned_lines = [clean_post_text(line) for line in lines]
                cleaned_lines = [l for l in cleaned_lines if l]
                new_text = '\n'.join(cleaned_lines)
            else:
                new_text = clean_post_text(post.transcript)

            if new_text and new_text != post.transcript:
                post.transcript = new_text
                post.title = new_text[:200]
                post.word_count = len(new_text.split())
                post.is_processed = False
                cleaned += 1

    print(f"Cleaned: {cleaned} posts")
    print(f"Skipped (no HTML): {skipped}")
    print("Done.")


if __name__ == '__main__':
    main()
