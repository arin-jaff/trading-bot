"""Tests for the prediction engine."""

import pytest
import math


class TestKellyCriterion:
    """Test Kelly criterion position sizing."""

    def test_favorable_yes_bet(self):
        from src.ml.predictor import TermPredictor
        predictor = TermPredictor()
        # If we think probability is 70% and price is 50%, Kelly should be positive
        kelly = predictor._kelly_criterion(0.7, 0.5)
        assert kelly > 0

    def test_unfavorable_bet(self):
        from src.ml.predictor import TermPredictor
        predictor = TermPredictor()
        # If we think probability equals price, Kelly should be ~0
        kelly = predictor._kelly_criterion(0.5, 0.5)
        assert kelly == 0 or kelly < 0.01

    def test_max_kelly_cap(self):
        from src.ml.predictor import TermPredictor
        predictor = TermPredictor()
        # Even with huge edge, Kelly should be capped at 25%
        kelly = predictor._kelly_criterion(0.99, 0.01)
        assert kelly <= 0.25

    def test_no_bet_on_no_edge(self):
        from src.ml.predictor import TermPredictor
        predictor = TermPredictor()
        kelly = predictor._kelly_criterion(0.3, 0.5)
        assert kelly >= 0  # Should be a NO bet scenario


class TestTrendScore:
    """Test trend score calculation."""

    def test_sigmoid_mapping(self):
        from src.ml.predictor import TermPredictor
        predictor = TermPredictor()

        class MockTerm:
            trend_score = 0.0

        term = MockTerm()

        # Zero trend -> 0.5
        term.trend_score = 0.0
        score = predictor._trend_score(None, term)
        assert abs(score - 0.5) < 0.01

        # Positive trend -> > 0.5
        term.trend_score = 2.0
        score = predictor._trend_score(None, term)
        assert score > 0.5

        # Negative trend -> < 0.5
        term.trend_score = -2.0
        score = predictor._trend_score(None, term)
        assert score < 0.5
