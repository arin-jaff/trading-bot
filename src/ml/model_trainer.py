"""Train and manage ML models for term prediction."""

import os
import pickle
import json
from datetime import datetime
from typing import Optional
from loguru import logger

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import CalibratedClassifierCV

from .feature_engineering import FeatureEngineering


MODEL_DIR = os.path.join('data', 'models')


class ModelTrainer:
    """Trains, evaluates, and manages prediction models."""

    def __init__(self):
        self.feature_eng = FeatureEngineering()
        self.models = {}
        self.scaler = StandardScaler()
        os.makedirs(MODEL_DIR, exist_ok=True)

    def train(self, retrain: bool = False) -> dict:
        """Train all models on historical data.

        Returns training metrics.
        """
        X, y = self.feature_eng.build_training_data()

        if X.empty or len(y) < 10:
            logger.warning(f"Insufficient training data: {len(y)} samples")
            return {'error': 'insufficient data', 'samples': len(y)}

        # Handle missing values
        X = X.fillna(0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Define models
        model_configs = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                C=0.1,
                random_state=42,
            ),
        }

        results = {}

        for name, model in model_configs.items():
            logger.info(f"Training {name}...")

            # Time-series aware cross-validation
            tscv = TimeSeriesSplit(n_splits=min(5, len(y) // 5))

            try:
                cv_scores = cross_val_score(
                    model, X_scaled, y, cv=tscv, scoring='roc_auc'
                )

                # Train on full data
                model.fit(X_scaled, y)

                # Calibrate probabilities
                calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
                try:
                    calibrated.fit(X_scaled, y)
                    self.models[name] = calibrated
                except Exception:
                    self.models[name] = model

                # Evaluate
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                results[name] = {
                    'cv_auc_mean': float(np.mean(cv_scores)),
                    'cv_auc_std': float(np.std(cv_scores)),
                    'train_accuracy': float(accuracy_score(y, y_pred)),
                    'train_precision': float(precision_score(y, y_pred, zero_division=0)),
                    'train_recall': float(recall_score(y, y_pred, zero_division=0)),
                    'train_f1': float(f1_score(y, y_pred, zero_division=0)),
                    'train_brier': float(brier_score_loss(y, y_prob)),
                    'n_samples': len(y),
                    'n_features': X.shape[1],
                }

                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X.columns, model.feature_importances_))
                    results[name]['top_features'] = dict(
                        sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    )

                logger.info(f"{name}: CV AUC = {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")

            except Exception as e:
                logger.error(f"Training {name} failed: {e}")
                results[name] = {'error': str(e)}

        # Save models
        self._save_models()

        # Save training metrics
        metrics_path = os.path.join(MODEL_DIR, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'results': results,
            }, f, indent=2)

        return results

    def predict(self, term_ids: list[int] = None) -> list[dict]:
        """Generate predictions using trained models.

        Returns predictions with ensemble averaging across models.
        """
        if not self.models:
            self._load_models()

        if not self.models:
            logger.warning("No trained models available")
            return []

        X = self.feature_eng.build_feature_matrix(term_ids)
        if X.empty:
            return []

        # Preserve term info
        term_info = X[['term_id', 'term']].copy()

        # Prepare features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].fillna(0)

        try:
            X_scaled = self.scaler.transform(X_numeric)
        except Exception:
            X_scaled = X_numeric.values

        predictions = []

        for idx in range(len(X)):
            x = X_scaled[idx:idx + 1]
            model_probs = []

            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(x)[0][1]
                    else:
                        prob = float(model.predict(x)[0])
                    model_probs.append((name, prob))
                except Exception as e:
                    logger.debug(f"Model {name} prediction error: {e}")

            if model_probs:
                # Weighted ensemble (weight by CV AUC if available)
                avg_prob = np.mean([p for _, p in model_probs])

                predictions.append({
                    'term_id': int(term_info.iloc[idx]['term_id']),
                    'term': term_info.iloc[idx]['term'],
                    'probability': float(avg_prob),
                    'model_probabilities': {n: float(p) for n, p in model_probs},
                    'model_name': 'ml_ensemble',
                    'confidence': self._prediction_confidence(model_probs),
                })

        return predictions

    def _prediction_confidence(self, model_probs: list[tuple]) -> float:
        """Calculate confidence based on model agreement."""
        if not model_probs:
            return 0

        probs = [p for _, p in model_probs]
        # Higher agreement (lower std) = higher confidence
        agreement = 1 - min(1, np.std(probs) * 4)
        # More models = higher confidence
        model_count_factor = min(1, len(probs) / 3)
        return float(agreement * model_count_factor)

    def _save_models(self):
        """Save trained models to disk."""
        for name, model in self.models.items():
            path = os.path.join(MODEL_DIR, f'{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)

        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Saved {len(self.models)} models to {MODEL_DIR}")

    def _load_models(self):
        """Load trained models from disk."""
        for name in ['gradient_boosting', 'random_forest', 'logistic_regression']:
            path = os.path.join(MODEL_DIR, f'{name}.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)

        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        logger.info(f"Loaded {len(self.models)} models")

    def get_model_info(self) -> dict:
        """Get information about trained models."""
        metrics_path = os.path.join(MODEL_DIR, 'training_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                return json.load(f)
        return {'status': 'no models trained'}
