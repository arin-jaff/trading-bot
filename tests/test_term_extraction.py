"""Tests for term extraction from Kalshi market titles."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kalshi.market_sync import MarketSync


@pytest.fixture
def sync():
    return MarketSync(client=None)


class TestTermExtraction:
    """Test that terms are correctly extracted from various market title formats."""

    def test_single_quoted_term(self, sync):
        market = {'title': "Will Trump say 'tariff'?", 'subtitle': ''}
        terms = sync.extract_terms_from_market(market)
        assert any(t['normalized_term'] == 'tariff' for t in terms)

    def test_double_quoted_term(self, sync):
        market = {'title': 'Will Trump mention "China"?', 'subtitle': ''}
        terms = sync.extract_terms_from_market(market)
        assert any(t['normalized_term'] == 'china' for t in terms)

    def test_multi_word_phrase(self, sync):
        market = {'title': "Will Trump say 'fake news'?", 'subtitle': ''}
        terms = sync.extract_terms_from_market(market)
        assert any(t['normalized_term'] == 'fake news' for t in terms)

    def test_compound_slash_term(self, sync):
        market = {
            'title': "Will Trump say 'who are you with / where are you from'?",
            'subtitle': ''
        }
        terms = sync.extract_terms_from_market(market)
        compound = [t for t in terms if t.get('is_compound')]
        assert len(compound) > 0 or any(
            'who are you with' in t['normalized_term'] for t in terms
        )

    def test_slash_in_title(self, sync):
        market = {
            'title': "Trump: 'great / tremendous'",
            'subtitle': ''
        }
        terms = sync.extract_terms_from_market(market)
        assert len(terms) > 0

    def test_multiple_quoted_terms(self, sync):
        market = {
            'title': "Will Trump say 'border' or 'wall'?",
            'subtitle': ''
        }
        terms = sync.extract_terms_from_market(market)
        normalized = [t['normalized_term'] for t in terms]
        assert 'border' in normalized
        assert 'wall' in normalized

    def test_subtitle_terms(self, sync):
        market = {
            'title': 'Trump Mentions Market',
            'subtitle': "Will he say 'incredible'?"
        }
        terms = sync.extract_terms_from_market(market)
        assert any(t['normalized_term'] == 'incredible' for t in terms)

    def test_term_normalization(self, sync):
        market = {'title': "Will Trump say 'MAGA'?", 'subtitle': ''}
        terms = sync.extract_terms_from_market(market)
        assert any(t['normalized_term'] == 'maga' for t in terms)
        # Original case preserved in 'term'
        assert any(t['term'] == 'MAGA' for t in terms)


class TestTermAnalysis:
    """Test term counting in transcripts."""

    def test_basic_counting(self):
        from src.scraper.term_analyzer import TermAnalyzer
        analyzer = TermAnalyzer()
        count, snippets = analyzer._count_term(
            "the tariff will be great. tariff is good. tariffs are coming.",
            "tariff"
        )
        # Should match "tariff" but not "tariffs" (word boundary)
        assert count == 2

    def test_case_insensitive(self):
        from src.scraper.term_analyzer import TermAnalyzer
        analyzer = TermAnalyzer()
        count, _ = analyzer._count_term(
            "China is great. CHINA will pay. china trade.",
            "china"
        )
        assert count == 3

    def test_phrase_counting(self):
        from src.scraper.term_analyzer import TermAnalyzer
        analyzer = TermAnalyzer()
        count, _ = analyzer._count_term(
            "I love this country. fake news is terrible. the fake news media is bad.",
            "fake news"
        )
        assert count == 2

    def test_context_snippets(self):
        from src.scraper.term_analyzer import TermAnalyzer
        analyzer = TermAnalyzer()
        _, snippets = analyzer._count_term(
            "we will impose a tariff on every country that cheats us",
            "tariff"
        )
        assert len(snippets) == 1
        assert 'tariff' in snippets[0]
