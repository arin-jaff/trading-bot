"""Analyzes speech transcripts for term occurrences and patterns."""

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from sqlalchemy import func

from ..database.models import Speech, Term, TermOccurrence
from ..database.db import get_session


class TermAnalyzer:
    """Processes speech transcripts to track term usage over time."""

    def process_all_unprocessed(self) -> int:
        """Process all speeches that haven't been analyzed yet.

        Each speech is committed individually to avoid holding the DB
        write lock for the entire batch (which could be 100K+ inserts).
        """
        # First, get the IDs and terms in a read-only query
        with get_session() as session:
            speech_ids = [
                s.id for s in session.query(Speech.id).filter_by(is_processed=False).filter(
                    Speech.transcript.isnot(None),
                    Speech.transcript != ''
                ).all()
            ]
            terms_data = [
                {'id': t.id, 'normalized_term': t.normalized_term,
                 'is_compound': t.is_compound, 'sub_terms': t.sub_terms}
                for t in session.query(Term).all()
            ]

        if not terms_data:
            logger.warning("No terms in database. Run market sync first.")
            return 0

        # Process each speech in its own short transaction
        count = 0
        for speech_id in speech_ids:
            try:
                with get_session() as session:
                    speech = session.query(Speech).get(speech_id)
                    if not speech or speech.is_processed:
                        continue
                    terms = session.query(Term).all()
                    self._process_speech(session, speech, terms)
                    speech.is_processed = True
                count += 1
            except Exception as e:
                logger.error(f"Error processing speech {speech_id}: {e}")

        # Update aggregate counts in a separate short transaction
        if count > 0:
            try:
                with get_session() as session:
                    self._update_term_stats(session)
            except Exception as e:
                logger.error(f"Error updating term stats: {e}")

        logger.info(f"Processed {count} speeches")
        return count

    def _process_speech(self, session, speech: Speech, terms: list[Term]):
        """Analyze a single speech for all tracked terms."""
        text = speech.transcript.lower()

        for term in terms:
            count = 0
            snippets = []

            if term.is_compound and term.sub_terms:
                # For compound terms like "X / Y", count if ANY sub-term appears
                for sub in term.sub_terms:
                    sub_count, sub_snippets = self._count_term(text, sub)
                    count += sub_count
                    snippets.extend(sub_snippets)
            else:
                count, snippets = self._count_term(text, term.normalized_term)

            if count > 0:
                # Check if occurrence already exists
                existing = session.query(TermOccurrence).filter_by(
                    term_id=term.id, speech_id=speech.id
                ).first()

                if not existing:
                    occ = TermOccurrence(
                        term_id=term.id,
                        speech_id=speech.id,
                        count=count,
                        context_snippets=snippets[:10],  # Keep top 10 snippets
                    )
                    session.add(occ)

    def _count_term(self, text: str, term: str) -> tuple[int, list[str]]:
        """Count occurrences of a term in text and extract context snippets."""
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(term) + r'\b'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        count = len(matches)

        snippets = []
        for match in matches[:10]:
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            snippet = '...' + text[start:end].strip() + '...'
            snippets.append(snippet)

        return count, snippets

    def _update_term_stats(self, session):
        """Update aggregate statistics for all terms."""
        terms = session.query(Term).all()
        for term in terms:
            total = session.query(func.sum(TermOccurrence.count)).filter_by(
                term_id=term.id
            ).scalar() or 0
            term.total_occurrences = total

            # Calculate trend score based on recent vs historical usage
            term.trend_score = self._calculate_trend(session, term.id)

    def _calculate_trend(self, session, term_id: int) -> float:
        """Calculate a trend score: positive = increasing usage, negative = decreasing.

        Compares last 30 days to previous 30 days.
        """
        now = datetime.utcnow()
        recent_start = now - timedelta(days=30)
        older_start = now - timedelta(days=60)

        recent_count = session.query(func.sum(TermOccurrence.count)).join(Speech).filter(
            TermOccurrence.term_id == term_id,
            Speech.date >= recent_start
        ).scalar() or 0

        older_count = session.query(func.sum(TermOccurrence.count)).join(Speech).filter(
            TermOccurrence.term_id == term_id,
            Speech.date >= older_start,
            Speech.date < recent_start
        ).scalar() or 0

        if older_count == 0:
            return float(recent_count)
        return (recent_count - older_count) / older_count

    def get_term_frequency_report(self) -> list[dict]:
        """Get frequency report for all terms across speeches."""
        with get_session() as session:
            terms = session.query(Term).order_by(Term.total_occurrences.desc()).all()
            report = []
            for term in terms:
                recent_occs = session.query(TermOccurrence).join(Speech).filter(
                    TermOccurrence.term_id == term.id
                ).order_by(Speech.date.desc()).limit(5).all()

                report.append({
                    'term': term.term,
                    'normalized': term.normalized_term,
                    'is_compound': term.is_compound,
                    'total_occurrences': term.total_occurrences,
                    'trend_score': term.trend_score,
                    'recent_speeches': [
                        {
                            'speech_title': occ.speech.title,
                            'date': occ.speech.date.isoformat() if occ.speech.date else None,
                            'count': occ.count,
                        }
                        for occ in recent_occs
                    ],
                })

            return report

    def get_term_time_series(self, term_id: int,
                             days: int = 365) -> list[dict]:
        """Get daily term usage over time for plotting."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        with get_session() as session:
            occs = session.query(
                func.date(Speech.date).label('day'),
                func.sum(TermOccurrence.count).label('total')
            ).join(Speech).filter(
                TermOccurrence.term_id == term_id,
                Speech.date >= cutoff
            ).group_by(func.date(Speech.date)).order_by('day').all()

            return [{'date': str(row.day), 'count': row.total} for row in occs]
