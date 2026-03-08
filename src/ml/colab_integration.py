"""Integration layer for Colab-trained model predictions.

Supports two modes:
1. File-based: Import predictions JSON exported from Colab notebooks
2. API-based: Call a live Colab-hosted model via ngrok tunnel

Replaces the OpenAI/Anthropic LLM calls with your own fine-tuned model.
"""

import os
import json
import glob
from datetime import datetime
from typing import Optional
from loguru import logger

import requests

from ..database.models import Term, TermPrediction
from ..database.db import get_session


class ColabPredictor:
    """Loads and serves predictions from Colab-trained Monte Carlo model."""

    def __init__(self):
        self.predictions_dir = os.path.join('data', 'predictions')
        self.colab_api_url = os.getenv('COLAB_PREDICTION_URL', '')
        self._cached_predictions = None
        self._cache_timestamp = None
        os.makedirs(self.predictions_dir, exist_ok=True)

    def get_predictions(self, force_refresh: bool = False) -> list[dict]:
        """Get the latest predictions, from file or API.

        Priority:
        1. Live Colab API (if configured and reachable)
        2. Latest predictions JSON file
        3. Empty list
        """
        # Try live API first
        if self.colab_api_url:
            try:
                preds = self._fetch_from_api()
                if preds:
                    return preds
            except Exception as e:
                logger.debug(f"Colab API unavailable: {e}")

        # Fall back to file-based predictions
        return self._load_from_file()

    def _fetch_from_api(self, event_context: Optional[dict] = None) -> list[dict]:
        """Fetch predictions from live Colab API."""
        url = self.colab_api_url.rstrip('/')

        # Check health
        try:
            health = requests.get(f'{url}/health', timeout=5)
            if health.status_code != 200:
                return []
        except Exception:
            return []

        # Get tracked terms
        with get_session() as session:
            terms = session.query(Term).all()
            term_list = [t.normalized_term for t in terms]

        # Request predictions
        payload = {
            'terms': term_list,
            'num_simulations': 100,  # lighter for live API
        }
        if event_context:
            payload.update(event_context)

        resp = requests.post(f'{url}/predict', json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        predictions = []
        for term_str, probability in data.get('predictions', {}).items():
            predictions.append({
                'term': term_str,
                'probability': probability,
                'confidence': min(1.0, data.get('num_simulations', 100) / 1000),
                'model_name': 'colab_monte_carlo_live',
            })

        logger.info(f"Fetched {len(predictions)} predictions from Colab API")
        return predictions

    def _load_from_file(self) -> list[dict]:
        """Load predictions from the latest JSON file."""
        # Check for predictions_latest.json first
        latest_path = os.path.join(self.predictions_dir, 'predictions_latest.json')

        if not os.path.exists(latest_path):
            # Try to find any predictions file
            pattern = os.path.join(self.predictions_dir, 'predictions_*.json')
            files = sorted(glob.glob(pattern), reverse=True)
            if files:
                latest_path = files[0]
            else:
                logger.warning("No prediction files found. Run the Colab notebook first.")
                return []

        try:
            with open(latest_path) as f:
                data = json.load(f)

            predictions = data.get('term_predictions', [])

            # Add term_id mapping
            with get_session() as session:
                term_map = {}
                for term in session.query(Term).all():
                    term_map[term.normalized_term] = term.id

                for pred in predictions:
                    normalized = pred['term'].lower().strip()
                    pred['term_id'] = term_map.get(normalized)

            logger.info(f"Loaded {len(predictions)} predictions from {latest_path}")
            logger.info(f"Generated at: {data.get('generated_at', 'unknown')}")

            # Also include discovered phrases as bonus intel
            discovered = data.get('discovered_phrases', [])
            if discovered:
                logger.info(f"Also found {len(discovered)} new phrases not in Kalshi list")

            return predictions

        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")
            return []

    def import_predictions_file(self, file_path: str) -> dict:
        """Import a predictions JSON file (downloaded from Colab/Drive).

        Copies it to the predictions directory and updates the latest symlink.
        """
        import shutil

        if not os.path.exists(file_path):
            return {'error': f'File not found: {file_path}'}

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Validate structure
            if 'term_predictions' not in data:
                return {'error': 'Invalid format: missing term_predictions'}

            # Copy to predictions dir
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            dest = os.path.join(self.predictions_dir, f'predictions_{timestamp}.json')
            shutil.copy2(file_path, dest)

            # Update latest
            latest = os.path.join(self.predictions_dir, 'predictions_latest.json')
            shutil.copy2(file_path, latest)

            stats = {
                'imported': True,
                'predictions_count': len(data['term_predictions']),
                'discovered_phrases': len(data.get('discovered_phrases', [])),
                'generated_at': data.get('generated_at', 'unknown'),
                'event': data.get('event_context', {}),
                'saved_to': dest,
            }

            logger.info(f"Imported predictions: {stats}")
            return stats

        except Exception as e:
            return {'error': str(e)}

    def save_to_database(self, predictions: Optional[list] = None,
                          event_id: Optional[int] = None):
        """Save Colab predictions to the database for the trading bot to use."""
        if predictions is None:
            predictions = self.get_predictions()

        if not predictions:
            logger.warning("No predictions to save")
            return

        with get_session() as session:
            saved = 0
            for pred in predictions:
                term_id = pred.get('term_id')
                if not term_id:
                    # Try to look up by name
                    term = session.query(Term).filter_by(
                        normalized_term=pred['term'].lower().strip()
                    ).first()
                    if term:
                        term_id = term.id
                    else:
                        continue

                tp = TermPrediction(
                    term_id=term_id,
                    event_id=event_id,
                    model_name=pred.get('model_name', 'colab_monte_carlo'),
                    probability=pred['probability'],
                    confidence=pred.get('confidence', 0.8),
                    reasoning=f"Monte Carlo simulation: {pred.get('speeches_containing', '?')}/{pred.get('total_mentions', '?')} speeches",
                    features_used={
                        'speeches_containing': pred.get('speeches_containing'),
                        'total_mentions': pred.get('total_mentions'),
                        'avg_mentions_per_speech': pred.get('avg_mentions_per_speech'),
                    },
                    target_date=datetime.utcnow(),
                )
                session.add(tp)
                saved += 1

        logger.info(f"Saved {saved} Colab predictions to database")

    def get_discovered_phrases(self) -> list[dict]:
        """Get phrases discovered by Monte Carlo that aren't in Kalshi's term list.

        These are potential future market terms — useful for anticipating
        new Kalshi markets before they're created.
        """
        latest_path = os.path.join(self.predictions_dir, 'predictions_latest.json')
        if not os.path.exists(latest_path):
            return []

        with open(latest_path) as f:
            data = json.load(f)

        return data.get('discovered_phrases', [])
