"""Markov chain text generator + Monte Carlo simulator for Pi-local training.

Replaces Colab's LoRA fine-tuned Llama with a lightweight word-level Markov
chain that trains in seconds and generates simulated Trump speeches for
term-frequency estimation.
"""

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


class MarkovChainTrainer:
    """Trains a word-level Markov chain on Trump speeches and runs Monte Carlo."""

    def __init__(self, order: int = 3):
        self.order = order
        self.chain = None  # dict[tuple[str,...], Counter[str]]
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
                corpus_word_count = sum(s.word_count or 0 for s in speeches)

            logger.info(f"Training Markov chain (order={self.order}) on {corpus_size} speeches")
            self._status.update(stage='Tokenizing corpus', progress=0.2)

            # Build chain
            chain = defaultdict(Counter)
            for i, transcript in enumerate(transcripts):
                words = self._tokenize(transcript)
                if len(words) < self.order + 1:
                    continue

                for j in range(len(words) - self.order):
                    key = tuple(words[j:j + self.order])
                    next_word = words[j + self.order]
                    chain[key][next_word] += 1

                if (i + 1) % 50 == 0:
                    self._status['progress'] = 0.2 + 0.5 * (i / len(transcripts))

            self.chain = dict(chain)
            self._status.update(stage='Saving model', progress=0.8)

            # Determine version
            version_str = self._next_version()

            # Save model artifact
            artifact_path = os.path.join(self.models_dir, f'markov_v{version_str}.pkl')
            with open(artifact_path, 'wb') as f:
                pickle.dump({
                    'chain': self.chain,
                    'order': self.order,
                    'version': version_str,
                    'trained_at': datetime.utcnow().isoformat(),
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
                f"({corpus_size} speeches, {len(self.chain)} states)"
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
                        word_count: Optional[int] = None) -> str:
        """Generate a simulated Trump speech using the Markov chain."""
        if not self.chain:
            self._load_latest_model()
        if not self.chain:
            return ""

        target_words = word_count or SCENARIO_WORD_COUNTS.get(scenario_type, 3000)

        # Pick a random starting state
        keys = list(self.chain.keys())
        if not keys:
            return ""

        current = random.choice(keys)
        words = list(current)

        for _ in range(target_words - self.order):
            key = tuple(words[-self.order:])
            if key not in self.chain:
                # Dead end — restart from random state
                current = random.choice(keys)
                words.extend(list(current))
                continue

            next_words = self.chain[key]
            # Weighted random selection
            total = sum(next_words.values())
            r = random.randint(1, total)
            cumulative = 0
            for word, count in next_words.items():
                cumulative += count
                if r <= cumulative:
                    words.append(word)
                    break

        return ' '.join(words)

    def run_monte_carlo(self, terms: list[str],
                        num_simulations: int = 2000,
                        scenario_weights: Optional[dict] = None) -> dict:
        """Run Monte Carlo simulation and compute term probabilities.

        Returns predictions dict in same format as Colab's predictions_latest.json.
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
                speech = self.generate_speech(scenario, word_count)
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

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize transcript into lowercase words."""
        # Remove bracketed stage directions [applause], (laughter), etc.
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        # Remove speaker labels like "TRUMP:" or "REPORTER:"
        text = re.sub(r'^[A-Z][A-Z\s]+:', '', text, flags=re.MULTILINE)
        # Split on whitespace, keep only alphabetic + common punctuation
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
        return words

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
                with open(latest.artifact_path, 'rb') as f:
                    data = pickle.load(f)
                self.chain = data['chain']
                self.order = data['order']
                logger.info(f"Loaded Markov chain v{latest.version}")
                return

        # Fallback: find any pickle file
        import glob
        files = sorted(glob.glob(os.path.join(self.models_dir, 'markov_v*.pkl')), reverse=True)
        if files:
            with open(files[0], 'rb') as f:
                data = pickle.load(f)
            self.chain = data['chain']
            self.order = data['order']
            logger.info(f"Loaded Markov chain from {files[0]}")
        else:
            logger.warning("No trained Markov model found — need to train first")
