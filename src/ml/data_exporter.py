"""Export speech data for fine-tuning in Google Colab.

Prepares training datasets from scraped transcripts in formats
compatible with LoRA fine-tuning pipelines (Unsloth, HuggingFace TRL).
"""

import json
import os
import re
from datetime import datetime
from typing import Optional
from loguru import logger

from ..database.models import Speech, Term, TermOccurrence
from ..database.db import get_session


EXPORT_DIR = os.path.join('data', 'exports')


class DataExporter:
    """Exports speech corpus and context data for Colab training."""

    def __init__(self):
        os.makedirs(EXPORT_DIR, exist_ok=True)

    def export_training_corpus(self, output_name: str = 'trump_speeches',
                                min_word_count: int = 100,
                                format: str = 'jsonl') -> dict:
        """Export all transcripts as a fine-tuning dataset.

        Formats:
        - 'jsonl': One JSON object per line, HuggingFace compatible
        - 'completion': OpenAI-style completion format
        - 'chat': Chat/instruction format for instruct models

        Returns export stats.
        """
        with get_session() as session:
            speeches = session.query(Speech).filter(
                Speech.transcript.isnot(None),
                Speech.word_count >= min_word_count
            ).order_by(Speech.date).all()

            if not speeches:
                logger.warning("No speeches with transcripts to export")
                return {'error': 'no data', 'count': 0}

            records = []
            for speech in speeches:
                transcript = self._clean_transcript(speech.transcript)
                if not transcript or len(transcript.split()) < min_word_count:
                    continue

                # Split long transcripts into chunks for training
                chunks = self._chunk_transcript(transcript, max_tokens=2048)

                for i, chunk in enumerate(chunks):
                    record = self._format_record(
                        speech, chunk, i, len(chunks), format
                    )
                    if record:
                        records.append(record)

            # Write output
            output_path = os.path.join(EXPORT_DIR, f'{output_name}.jsonl')
            with open(output_path, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')

            stats = {
                'total_speeches': len(speeches),
                'total_chunks': len(records),
                'output_path': output_path,
                'total_words': sum(
                    len(r.get('text', r.get('completion', '')).split())
                    for r in records
                ),
                'format': format,
            }

            # Also export metadata
            meta_path = os.path.join(EXPORT_DIR, f'{output_name}_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'export_date': datetime.utcnow().isoformat(),
                    'stats': stats,
                    'speeches': [
                        {
                            'id': s.id,
                            'title': s.title,
                            'date': s.date.isoformat() if s.date else None,
                            'type': s.speech_type,
                            'source': s.source,
                            'word_count': s.word_count,
                        }
                        for s in speeches
                    ],
                }, f, indent=2)

            logger.info(f"Exported {stats['total_chunks']} chunks from "
                       f"{stats['total_speeches']} speeches to {output_path}")
            return stats

    def export_term_context(self, output_name: str = 'term_context') -> dict:
        """Export term occurrence data with context snippets for RAG.

        This creates the historical retrieval database that the RAG
        pipeline will search during inference.
        """
        with get_session() as session:
            terms = session.query(Term).all()

            term_data = []
            for term in terms:
                occs = session.query(TermOccurrence).filter_by(
                    term_id=term.id
                ).all()

                contexts = []
                for occ in occs:
                    speech = session.query(Speech).get(occ.speech_id)
                    if speech and occ.context_snippets:
                        for snippet in occ.context_snippets:
                            contexts.append({
                                'snippet': snippet,
                                'speech_title': speech.title,
                                'speech_date': speech.date.isoformat() if speech.date else None,
                                'speech_type': speech.speech_type,
                                'count_in_speech': occ.count,
                            })

                term_data.append({
                    'term': term.term,
                    'normalized': term.normalized_term,
                    'is_compound': term.is_compound,
                    'sub_terms': term.sub_terms,
                    'total_occurrences': term.total_occurrences,
                    'trend_score': term.trend_score,
                    'contexts': contexts,
                })

            output_path = os.path.join(EXPORT_DIR, f'{output_name}.json')
            with open(output_path, 'w') as f:
                json.dump(term_data, f, indent=2)

            logger.info(f"Exported context for {len(term_data)} terms")
            return {
                'terms_exported': len(term_data),
                'total_contexts': sum(len(t['contexts']) for t in term_data),
                'output_path': output_path,
            }

    def export_event_history(self, output_name: str = 'event_speech_pairs') -> dict:
        """Export event-speech pairs for event-conditioned generation.

        Maps event metadata (type, location, audience) to what was actually said.
        Used by the RAG pipeline to retrieve relevant historical examples.
        """
        with get_session() as session:
            speeches = session.query(Speech).filter(
                Speech.transcript.isnot(None),
                Speech.speech_type.isnot(None)
            ).order_by(Speech.date).all()

            pairs = []
            for speech in speeches:
                transcript = self._clean_transcript(speech.transcript)
                if not transcript:
                    continue

                # Extract key phrases from this speech
                key_phrases = self._extract_key_phrases(transcript)

                pairs.append({
                    'speech_type': speech.speech_type,
                    'title': speech.title,
                    'date': speech.date.isoformat() if speech.date else None,
                    'source': speech.source,
                    'word_count': speech.word_count,
                    'key_phrases': key_phrases,
                    'first_500_words': ' '.join(transcript.split()[:500]),
                })

            output_path = os.path.join(EXPORT_DIR, f'{output_name}.json')
            with open(output_path, 'w') as f:
                json.dump(pairs, f, indent=2)

            return {
                'pairs_exported': len(pairs),
                'output_path': output_path,
            }

    def _clean_transcript(self, text: str) -> str:
        """Clean a transcript for training."""
        if not text:
            return ''

        # Remove common artifacts
        text = re.sub(r'\[.*?\]', '', text)  # [applause], [laughter], etc.
        text = re.sub(r'\(.*?\)', '', text)  # (crosstalk), etc.
        text = re.sub(r'TRUMP:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'PRESIDENT TRUMP:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'DONALD TRUMP:', '', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _chunk_transcript(self, text: str, max_tokens: int = 2048) -> list[str]:
        """Split a transcript into training chunks.

        Tries to split on sentence boundaries near the max_tokens limit.
        Overlaps slightly for context continuity.
        """
        words = text.split()
        if len(words) <= max_tokens:
            return [text]

        chunks = []
        overlap = 50  # words of overlap between chunks

        i = 0
        while i < len(words):
            end = min(i + max_tokens, len(words))
            chunk_words = words[i:end]
            chunk = ' '.join(chunk_words)

            # Try to end on a sentence boundary
            last_period = chunk.rfind('.')
            last_question = chunk.rfind('?')
            last_excl = chunk.rfind('!')
            best_break = max(last_period, last_question, last_excl)

            if best_break > len(chunk) * 0.7:  # Only if past 70% of chunk
                chunk = chunk[:best_break + 1]

            chunks.append(chunk)
            i = end - overlap

        return chunks

    def _format_record(self, speech: Speech, text: str,
                       chunk_idx: int, total_chunks: int,
                       format: str) -> Optional[dict]:
        """Format a training record based on the chosen format."""
        if format == 'jsonl':
            return {
                'text': text,
                'metadata': {
                    'speech_id': speech.id,
                    'title': speech.title,
                    'date': speech.date.isoformat() if speech.date else None,
                    'type': speech.speech_type,
                    'chunk': f'{chunk_idx + 1}/{total_chunks}',
                },
            }
        elif format == 'completion':
            # For causal LM fine-tuning
            return {
                'prompt': '',
                'completion': text,
            }
        elif format == 'chat':
            # Instruction format
            context = f"Event: {speech.speech_type or 'speech'}"
            if speech.title:
                context += f" | {speech.title}"
            if speech.date:
                context += f" | {speech.date.strftime('%B %d, %Y')}"

            return {
                'instruction': f"Give a speech as if you are Donald Trump at this event. {context}",
                'input': '',
                'output': text,
            }

        return None

    def _extract_key_phrases(self, text: str, top_n: int = 20) -> list[str]:
        """Extract key phrases from a text using simple n-gram frequency."""
        from collections import Counter

        words = text.lower().split()
        # Common stopwords to skip
        stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'and', 'or', 'but', 'not', 'that', 'this', 'it', 'they',
                 'we', 'you', 'he', 'she', 'i', 'me', 'my', 'your', 'his',
                 'her', 'our', 'their', 'its', 'so', 'very', 'just', 'about',
                 'up', 'out', 'if', 'than', 'them', 'then', 'what', 'which',
                 'who', 'whom', 'when', 'where', 'how', 'all', 'each',
                 'every', 'both', 'few', 'more', 'most', 'other', 'some',
                 'such', 'no', 'nor', 'only', 'own', 'same', 'too'}

        # 2-grams and 3-grams
        bigrams = Counter()
        trigrams = Counter()

        for i in range(len(words) - 1):
            if words[i] not in stops and words[i + 1] not in stops:
                bigrams[f"{words[i]} {words[i + 1]}"] += 1

        for i in range(len(words) - 2):
            w = (words[i], words[i + 1], words[i + 2])
            if w[0] not in stops or w[2] not in stops:
                trigrams[f"{w[0]} {w[1]} {w[2]}"] += 1

        # Combine and rank
        all_phrases = {}
        for phrase, count in bigrams.most_common(top_n * 2):
            if count >= 2:
                all_phrases[phrase] = count
        for phrase, count in trigrams.most_common(top_n * 2):
            if count >= 2:
                all_phrases[phrase] = count * 1.5  # weight trigrams higher

        sorted_phrases = sorted(all_phrases.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases[:top_n]]
