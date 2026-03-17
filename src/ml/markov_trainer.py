"""Markov chain text generator + Monte Carlo simulator for Pi-local training.

Replaces Colab's LoRA fine-tuned Llama with a lightweight word-level Markov
chain that trains in seconds and generates simulated Trump speeches for
term-frequency estimation.
"""

import math
import os
import re
import json
import time
import pickle
import random
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional
from loguru import logger

from ..database.db import get_session
from ..database.models import Speech, Term, ModelVersion


# Word counts per scenario type (approximate real-world lengths)
SCENARIO_WORD_COUNTS = {
    'rally': 5000,
    'press_conference': 2000,
    'chopper_talk': 800,
    'fox_interview': 1500,
    'social_media': 300,
}

DEFAULT_SCENARIO_WEIGHTS = {
    'rally': 0.40,
    'press_conference': 0.25,
    'chopper_talk': 0.10,
    'fox_interview': 0.15,
    'social_media': 0.10,
}

# Proper nouns to capitalize in post-processing
_PROPER_NOUNS = {
    'trump', 'donald', 'america', 'american', 'americans', 'usa',
    'china', 'chinese', 'russia', 'russian', 'ukraine', 'ukrainian',
    'biden', 'obama', 'kamala', 'harris', 'pelosi', 'schumer',
    'desantis', 'vivek', 'nikki', 'haley', 'pence', 'ivanka',
    'jared', 'kushner', 'melania', 'barron', 'eric', 'jr',
    'mexico', 'mexican', 'canada', 'canadian', 'iran', 'iranian',
    'north', 'korea', 'korean', 'japan', 'japanese', 'nato', 'eu',
    'congress', 'senate', 'republican', 'republicans', 'democrat',
    'democrats', 'gop', 'maga', 'florida', 'texas', 'california',
    'new', 'york', 'washington', 'jerusalem', 'israel', 'israeli',
    'taliban', 'isis', 'al', 'qaeda', 'afghanistan', 'iraq',
    'fbi', 'cia', 'doj', 'cnn', 'fox', 'msnbc', 'abc', 'nbc', 'cbs',
    'covid', 'wuhan', 'pfizer', 'moderna',
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'god', 'bible', 'christmas', 'easter',
}

# Current pickle format version
_FORMAT_VERSION = 2


class MarkovChainTrainer:
    """Trains a word-level Markov chain on Trump speeches and runs Monte Carlo."""

    def __init__(self, order: int = 3):
        self.order = order
        self.chain = None  # dict[tuple[str,...], Counter[str]]
        self.topic_vocab = {}  # dict[str, set[str]] — scenario → distinctive words
        self._word_to_states = {}  # dict[str, list[tuple]] — inverted index
        self.models_dir = os.path.join('data', 'models')
        self.predictions_dir = os.path.join('data', 'predictions')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)

        # Training status (for GUI polling)
        self._status = {
            'state': 'idle',  # idle, training, simulating, complete, error
            'stage': '',
            'progress': 0.0,
            'current_simulation': 0,
            'total_simulations': 0,
            'eta_seconds': None,
            'error': None,
        }

    def get_status(self) -> dict:
        return self._status.copy()

    def train(self) -> Optional[dict]:
        """Build Markov chain from all processed speech transcripts.

        Returns version info dict or None on failure.
        """
        start_time = time.time()
        self._status.update(state='training', stage='Loading speeches', progress=0.0)

        try:
            # Load transcripts
            with get_session() as session:
                speeches = session.query(Speech).filter(
                    Speech.transcript.isnot(None),
                    Speech.is_processed == True,
                    Speech.word_count >= 100,
                ).all()

                if not speeches:
                    logger.warning("No processed speeches found for training")
                    self._status.update(state='error', error='No speeches available')
                    return None

                corpus_size = len(speeches)
                transcripts = [s.transcript for s in speeches]
                speech_types = [s.speech_type for s in speeches]
                corpus_word_count = sum(s.word_count or 0 for s in speeches)

            logger.info(f"Training Markov chain (order={self.order}) on {corpus_size} speeches")
            self._status.update(stage='Tokenizing corpus', progress=0.2)

            # Build chain + per-type word frequencies for topic vocab
            chain = defaultdict(Counter)
            type_word_counts = defaultdict(Counter)  # speech_type → word → count
            corpus_word_freq = Counter()

            for i, transcript in enumerate(transcripts):
                words = self._tokenize(transcript)
                if len(words) < self.order + 1:
                    continue

                speech_type = speech_types[i] or 'unknown'
                word_only = [w for w in words if w not in '.!?,']
                type_word_counts[speech_type].update(word_only)
                corpus_word_freq.update(word_only)

                for j in range(len(words) - self.order):
                    key = tuple(words[j:j + self.order])
                    next_word = words[j + self.order]
                    chain[key][next_word] += 1

                if (i + 1) % 50 == 0:
                    self._status['progress'] = 0.2 + 0.5 * (i / len(transcripts))

            self.chain = dict(chain)

            # B: Build topic_vocab — words disproportionately common in each type
            self._status.update(stage='Building topic vocab', progress=0.72)
            self.topic_vocab = self._build_topic_vocab(
                type_word_counts, corpus_word_freq
            )

            # C: Build inverted index for fast bridge-state lookups
            self._status.update(stage='Building word index', progress=0.76)
            self._word_to_states = self._build_word_index(self.chain)

            self._status.update(stage='Saving model', progress=0.8)

            # Determine version
            version_str = self._next_version()

            # Save model artifact (format v2)
            artifact_path = os.path.join(self.models_dir, f'markov_v{version_str}.pkl')
            with open(artifact_path, 'wb') as f:
                pickle.dump({
                    'chain': self.chain,
                    'order': self.order,
                    'version': version_str,
                    'trained_at': datetime.utcnow().isoformat(),
                    'format_version': _FORMAT_VERSION,
                    'topic_vocab': {k: list(v) for k, v in self.topic_vocab.items()},
                    'word_to_states': self._word_to_states,
                }, f)

            training_duration = time.time() - start_time

            # Create ModelVersion record
            with get_session() as session:
                # Deactivate previous versions
                session.query(ModelVersion).filter_by(is_active=True).update(
                    {'is_active': False}
                )
                mv = ModelVersion(
                    version=version_str,
                    model_type='markov_chain',
                    markov_order=self.order,
                    corpus_size=corpus_size,
                    corpus_word_count=corpus_word_count,
                    training_duration_seconds=round(training_duration, 2),
                    artifact_path=artifact_path,
                    is_active=True,
                )
                session.add(mv)

            self._status.update(stage='Training complete', progress=1.0)
            logger.info(
                f"Markov chain v{version_str} trained in {training_duration:.1f}s "
                f"({corpus_size} speeches, {len(self.chain)} states, "
                f"topic vocabs: {{{', '.join(f'{k}: {len(v)}' for k, v in self.topic_vocab.items())}}})"
            )

            return {
                'version': version_str,
                'corpus_size': corpus_size,
                'corpus_word_count': corpus_word_count,
                'chain_states': len(self.chain),
                'training_seconds': round(training_duration, 2),
            }

        except Exception as e:
            self._status.update(state='error', error=str(e))
            logger.error(f"Markov training failed: {e}")
            return None

    def generate_speech(self, scenario_type: str = 'rally',
                        word_count: Optional[int] = None,
                        temperature: float = 1.0,
                        topic_bias: float = 1.5) -> str:
        """Generate a simulated Trump speech using the Markov chain.

        Args:
            scenario_type: Type of speech scenario for length and topic bias.
            word_count: Override target word count.
            temperature: Sampling temperature (0.3-2.0). 1.0 = default behavior.
            topic_bias: Multiplier for topic-relevant words (1.0 = no bias).
        """
        if not self.chain:
            self._load_latest_model()
        if not self.chain:
            return ""

        target_words = word_count or SCENARIO_WORD_COUNTS.get(scenario_type, 3000)
        topic_words = self.topic_vocab.get(scenario_type, set())

        # Pick a random starting state
        keys = list(self.chain.keys())
        if not keys:
            return ""

        current = random.choice(keys)
        words = list(current)

        for _ in range(target_words - self.order):
            key = tuple(words[-self.order:])
            if key not in self.chain:
                bridge = self._find_bridge_state(words)
                words.extend(list(bridge))
                continue

            next_word = self._sample_next_word(
                self.chain[key], temperature, topic_words, topic_bias
            )
            words.append(next_word)

        return self._post_process(' '.join(words))

    def _generate_raw(self, scenario_type: str = 'rally',
                      word_count: Optional[int] = None) -> str:
        """Generate raw text for Monte Carlo — no post-processing, temp=1.0, no topic bias.

        This preserves identical behavior to the original generate_speech() for
        prediction accuracy. Punctuation tokens are included but don't affect
        \\b word-boundary regex matching used by run_monte_carlo().
        """
        if not self.chain:
            self._load_latest_model()
        if not self.chain:
            return ""

        target_words = word_count or SCENARIO_WORD_COUNTS.get(scenario_type, 3000)

        keys = list(self.chain.keys())
        if not keys:
            return ""

        current = random.choice(keys)
        words = list(current)

        for _ in range(target_words - self.order):
            key = tuple(words[-self.order:])
            if key not in self.chain:
                bridge = self._find_bridge_state(words)
                words.extend(list(bridge))
                continue

            next_words = self.chain[key]
            # Original weighted random selection (temp=1.0, no bias)
            total = sum(next_words.values())
            r = random.randint(1, total)
            cumulative = 0
            for word, count in next_words.items():
                cumulative += count
                if r <= cumulative:
                    words.append(word)
                    break

        return ' '.join(words)

    def generate_from_prompt(self, prompt: str,
                            word_count: int = 500,
                            temperature: float = 1.0,
                            qa_mode: bool = False) -> str:
        """Generate text seeded from a user prompt.

        Tokenizes the prompt, finds the best matching chain state to continue
        from, then generates forward using the Markov chain.

        If qa_mode=True, strips question words and starts generation from
        topic keywords rather than echoing the question.
        """
        if not self.chain:
            self._load_latest_model()
        if not self.chain:
            return ""

        prompt_words = self._tokenize(prompt)
        keys = list(self.chain.keys())
        if not keys:
            return ""

        if qa_mode and prompt_words:
            # Q&A mode: extract topic keywords, skip question scaffolding
            topic_keywords = self._extract_topic_keywords(prompt_words)
            start_key = self._find_topic_start(topic_keywords, keys)
            words = list(start_key)
        else:
            # Normal continuation mode
            start_key = self._find_prompt_start(prompt_words, keys)
            words = list(prompt_words) if prompt_words else list(start_key)

        for _ in range(word_count):
            key = tuple(words[-self.order:])
            if key not in self.chain:
                bridge = self._find_bridge_state(words)
                words.extend(list(bridge))
                continue

            next_word = self._sample_next_word(self.chain[key], temperature)
            words.append(next_word)

        return self._post_process(' '.join(words))

    def run_monte_carlo(self, terms: list[str],
                        num_simulations: int = 2000,
                        scenario_weights: Optional[dict] = None) -> dict:
        """Run Monte Carlo simulation and compute term probabilities.

        Returns predictions dict in same format as Colab's predictions_latest.json.
        Uses _generate_raw() to preserve prediction behavior (no post-processing).
        """
        if not self.chain:
            self._load_latest_model()
        if not self.chain:
            return {}

        weights = scenario_weights or DEFAULT_SCENARIO_WEIGHTS
        self._status.update(
            state='simulating',
            stage='Running Monte Carlo',
            progress=0.0,
            current_simulation=0,
            total_simulations=num_simulations,
        )

        # Normalize terms for matching
        term_patterns = {}
        for term in terms:
            normalized = term.lower().strip()
            # Pre-compile regex for whole-word matching
            pattern = re.compile(r'\b' + re.escape(normalized) + r'\b', re.IGNORECASE)
            term_patterns[normalized] = pattern

        # Track per-term statistics
        term_stats = {t: {'speeches_containing': 0, 'total_mentions': 0}
                      for t in term_patterns}

        # Per-scenario simulation counts
        scenario_sims = {}
        for scenario, weight in weights.items():
            scenario_sims[scenario] = max(1, int(num_simulations * weight))

        start_time = time.time()
        sim_count = 0
        total_sims = sum(scenario_sims.values())
        total_words_generated = 0

        for scenario, n_sims in scenario_sims.items():
            word_count = SCENARIO_WORD_COUNTS.get(scenario, 3000)

            for i in range(n_sims):
                speech = self._generate_raw(scenario, word_count)
                total_words_generated += len(speech.split())

                # Count term occurrences
                for term_key, pattern in term_patterns.items():
                    matches = pattern.findall(speech)
                    if matches:
                        term_stats[term_key]['speeches_containing'] += 1
                        term_stats[term_key]['total_mentions'] += len(matches)

                sim_count += 1
                if sim_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = sim_count / elapsed if elapsed > 0 else 1
                    eta = (total_sims - sim_count) / rate if rate > 0 else 0
                    self._status.update(
                        progress=sim_count / total_sims,
                        current_simulation=sim_count,
                        total_simulations=total_sims,
                        eta_seconds=round(eta, 1),
                    )

        # Compute probabilities
        avg_words = total_words_generated / total_sims if total_sims > 0 else 3000
        predictions = []
        for term_key, stats in term_stats.items():
            probability = stats['speeches_containing'] / total_sims if total_sims > 0 else 0
            avg_mentions = stats['total_mentions'] / total_sims if total_sims > 0 else 0

            predictions.append({
                'term': term_key,
                'probability': round(probability, 4),
                'speeches_containing': stats['speeches_containing'],
                'total_mentions': stats['total_mentions'],
                'avg_mentions_per_speech': round(avg_mentions, 4),
                'model_name': 'markov_monte_carlo',
                'confidence': min(1.0, total_sims / 2000),
            })

        duration = time.time() - start_time
        self._status.update(
            state='complete',
            stage='Simulation complete',
            progress=1.0,
            current_simulation=total_sims,
        )

        logger.info(
            f"Monte Carlo complete: {total_sims} simulations in {duration:.1f}s, "
            f"{len(predictions)} terms evaluated"
        )

        return {
            'term_predictions': predictions,
            'simulation_params': {
                'num_simulations': total_sims,
                'scenario_weights': weights,
                'avg_words_per_speech': round(avg_words, 0),
                'model_type': 'markov_chain',
                'markov_order': self.order,
            },
            'generated_at': datetime.utcnow().isoformat(),
            'discovered_phrases': [],
        }

    def save_predictions(self, predictions_data: dict) -> str:
        """Write predictions to data/predictions/predictions_latest.json."""
        path = os.path.join(self.predictions_dir, 'predictions_latest.json')
        with open(path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        logger.info(f"Saved predictions to {path}")

        # Also save timestamped copy
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        ts_path = os.path.join(self.predictions_dir, f'predictions_{ts}.json')
        with open(ts_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        return path

    # ── A: Punctuation-aware tokenization ──

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize transcript into lowercase words + punctuation tokens."""
        # Remove bracketed stage directions [applause], (laughter), etc.
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        # Remove speaker labels like "TRUMP:" or "REPORTER:"
        text = re.sub(r'^[A-Z][A-Z\s]+:', '', text, flags=re.MULTILINE)
        # Split into words + punctuation tokens (.!?,)
        words = re.findall(r"[a-z]+(?:'[a-z]+)?|[.!?,]", text.lower())
        # Collapse consecutive punctuation (e.g. "..." → ".", "!!" → "!")
        result = []
        for w in words:
            if w in '.!?,' and result and result[-1] in '.!?,':
                continue  # skip duplicate punctuation
            result.append(w)
        return result

    # ── B: Topic vocab builder ──

    @staticmethod
    def _build_topic_vocab(type_word_counts: dict[str, Counter],
                           corpus_word_freq: Counter,
                           min_occurrences: int = 5,
                           min_ratio: float = 2.0) -> dict[str, set[str]]:
        """Build per-scenario topic vocabularies.

        A word is "distinctive" for a scenario if it appears at least
        min_occurrences times in that type AND its frequency ratio vs the
        corpus average is > min_ratio.
        """
        total_corpus = sum(corpus_word_freq.values()) or 1
        topic_vocab = {}

        for speech_type, wc in type_word_counts.items():
            type_total = sum(wc.values()) or 1
            distinctive = set()
            for word, count in wc.items():
                if count < min_occurrences:
                    continue
                type_freq = count / type_total
                corpus_freq = corpus_word_freq[word] / total_corpus
                if corpus_freq > 0 and type_freq / corpus_freq > min_ratio:
                    distinctive.add(word)
            topic_vocab[speech_type] = distinctive

        return topic_vocab

    # ── C: Inverted index + bridge-state lookup ──

    @staticmethod
    def _build_word_index(chain: dict) -> dict[str, list[tuple]]:
        """Build inverted index: word → list of chain states containing it."""
        index = defaultdict(list)
        for state in chain:
            for word in set(state):  # deduplicate within state
                if word not in '.!?,':
                    index[word].append(state)
        return dict(index)

    def _find_bridge_state(self, words: list[str]) -> tuple:
        """Find a contextually relevant bridge state when hitting a dead end.

        Tries recent words against the inverted index for O(1) lookup,
        falls back to random state.
        """
        # Look at the last few content words (skip punctuation)
        recent = [w for w in words[-10:] if w not in '.!?,']
        for w in reversed(recent):
            states = self._word_to_states.get(w)
            if states:
                return random.choice(states)

        # No inverted index or no match — random fallback
        keys = list(self.chain.keys())
        return random.choice(keys) if keys else ()

    # ── Q&A helpers ──

    # Words that form question scaffolding, not topic content
    _QUESTION_WORDS = frozenset({
        'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whom',
        'do', 'does', 'did', 'is', 'are', 'was', 'were', 'will', 'would',
        'can', 'could', 'should', 'shall', 'may', 'might', 'has', 'have',
        'had', 'the', 'a', 'an', 'about', 'your', 'you', 'think', 'tell',
        'us', 'me', 'of', 'on', 'in', 'to', 'for', 'with', 'opinion',
    })

    def _extract_topic_keywords(self, prompt_words: list[str]) -> list[str]:
        """Extract topic keywords from a question, filtering out question scaffolding."""
        keywords = [w for w in prompt_words
                    if w not in self._QUESTION_WORDS and w not in '.!?,']
        # If everything got filtered, fall back to all content words
        if not keywords:
            keywords = [w for w in prompt_words if w not in '.!?,']
        return keywords

    def _find_topic_start(self, topic_keywords: list[str], keys: list[tuple]) -> tuple:
        """Find the best chain state matching topic keywords."""
        # Try each keyword against the inverted index, prefer later (more specific) keywords
        for word in reversed(topic_keywords):
            states = self._word_to_states.get(word)
            if states:
                # Prefer states that contain multiple topic keywords
                scored = []
                for state in states:
                    overlap = sum(1 for kw in topic_keywords if kw in state)
                    scored.append((overlap, state))
                scored.sort(key=lambda x: x[0], reverse=True)
                # Pick randomly among top-scoring states
                top_score = scored[0][0]
                top_states = [s for score, s in scored if score == top_score]
                return random.choice(top_states)
        return random.choice(keys)

    def _find_prompt_start(self, prompt_words: list[str], keys: list[tuple]) -> tuple:
        """Find the best chain state to continue from a prompt."""
        if len(prompt_words) >= self.order:
            candidate = tuple(prompt_words[-self.order:])
            if candidate in self.chain:
                return candidate

        if prompt_words:
            last_word = prompt_words[-1]
            states = self._word_to_states.get(last_word)
            if states:
                return random.choice(states)
            matching = [k for k in keys if last_word in k]
            if matching:
                return random.choice(matching)

        return random.choice(keys)

    # ── D: Temperature-controlled sampling ──

    @staticmethod
    def _sample_next_word(next_words: Counter, temperature: float = 1.0,
                          topic_words: set = None,
                          topic_bias: float = 1.0) -> str:
        """Sample next word with temperature and optional topic bias.

        Args:
            next_words: Counter of {word: count} from chain.
            temperature: 1.0 = default, <1.0 = sharper/more coherent, >1.0 = flatter/creative.
            topic_words: Set of words distinctive to the current scenario.
            topic_bias: Multiplier applied to counts for topic-relevant words.
        """
        # Fast path: temp=1.0 and no topic bias — original integer arithmetic
        if temperature == 1.0 and (not topic_words or topic_bias == 1.0):
            total = sum(next_words.values())
            r = random.randint(1, total)
            cumulative = 0
            for word, count in next_words.items():
                cumulative += count
                if r <= cumulative:
                    return word
            return next(iter(next_words))  # shouldn't reach here

        # Apply topic bias and temperature
        adjusted = {}
        for word, count in next_words.items():
            weight = float(count)
            if topic_words and topic_bias != 1.0 and word in topic_words:
                weight *= topic_bias
            if temperature != 1.0:
                weight = weight ** (1.0 / temperature)
            adjusted[word] = weight

        total = sum(adjusted.values())
        if total <= 0:
            return random.choice(list(next_words.keys()))

        r = random.random() * total
        cumulative = 0.0
        for word, weight in adjusted.items():
            cumulative += weight
            if r <= cumulative:
                return word
        return next(iter(adjusted))  # rounding guard

    # ── E: Post-processing (capitalization + punctuation cleanup) ──

    @staticmethod
    def _post_process(text: str) -> str:
        """Clean up generated text for user-facing display.

        - Remove spaces before punctuation
        - Capitalize after sentence-enders and at start
        - Capitalize "i" → "I"
        - Capitalize known proper nouns
        """
        # Remove space before punctuation tokens
        text = re.sub(r'\s+([.!?,])', r'\1', text)
        # Ensure space after punctuation if followed by a letter
        text = re.sub(r'([.!?,])([a-zA-Z])', r'\1 \2', text)

        # Split into sentences and capitalize
        result = []
        capitalize_next = True
        for token in re.split(r'(\s+)', text):
            if not token.strip():
                result.append(token)
                continue
            if capitalize_next and token[0].isalpha():
                token = token[0].upper() + token[1:]
                capitalize_next = False
            if token[-1] in '.!?':
                capitalize_next = True
            result.append(token)

        text = ''.join(result)

        # Capitalize "i" as standalone word
        text = re.sub(r"\bi\b", "I", text)
        # Capitalize "i'" contractions (i'm, i've, i'll, i'd)
        text = re.sub(r"\bi'", "I'", text)

        # Capitalize proper nouns
        for noun in _PROPER_NOUNS:
            text = re.sub(
                r'\b' + re.escape(noun) + r'\b',
                lambda m: m.group().capitalize(),
                text,
                flags=re.IGNORECASE,
            )

        # Fix double-capitalized all-caps abbreviations that got lowered then capitalized
        for abbr in ('FBI', 'CIA', 'DOJ', 'CNN', 'MSNBC', 'ABC', 'NBC', 'CBS',
                      'USA', 'GOP', 'MAGA', 'NATO', 'EU', 'ISIS', 'COVID'):
            text = re.sub(r'\b' + abbr.capitalize() + r'\b', abbr, text)

        return text

    def _next_version(self) -> str:
        """Get next version string by incrementing patch number."""
        with get_session() as session:
            latest = session.query(ModelVersion).order_by(
                ModelVersion.created_at.desc()
            ).first()

        if not latest:
            return '1.0.0'

        parts = latest.version.split('.')
        if len(parts) == 3:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f'{major}.{minor}.{patch + 1}'

        return '1.0.0'

    def _load_latest_model(self):
        """Load the most recent trained Markov chain from disk."""
        with get_session() as session:
            latest = session.query(ModelVersion).filter_by(
                is_active=True, model_type='markov_chain'
            ).order_by(ModelVersion.created_at.desc()).first()

            if latest and latest.artifact_path and os.path.exists(latest.artifact_path):
                self._load_pickle(latest.artifact_path)
                logger.info(f"Loaded Markov chain v{latest.version}")
                return

        # Fallback: find any pickle file
        import glob
        files = sorted(glob.glob(os.path.join(self.models_dir, 'markov_v*.pkl')), reverse=True)
        if files:
            self._load_pickle(files[0])
            logger.info(f"Loaded Markov chain from {files[0]}")
        else:
            logger.warning("No trained Markov model found — need to train first")

    def _load_pickle(self, path: str):
        """Load a pickle file, handling both v1 and v2 formats."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.chain = data['chain']
        self.order = data['order']

        # v2 fields — gracefully degrade for old pickles
        fmt = data.get('format_version', 1)
        if fmt >= 2:
            raw_vocab = data.get('topic_vocab', {})
            self.topic_vocab = {k: set(v) for k, v in raw_vocab.items()}
            self._word_to_states = data.get('word_to_states', {})
        else:
            self.topic_vocab = {}
            self._word_to_states = {}
