"""Sync Kalshi Trump Mentions markets to local database."""

import re
from datetime import datetime
from typing import Optional
from loguru import logger
from sqlalchemy.orm import Session

from ..database.models import Market, Term, market_term_association
from ..database.db import get_session
from .client import KalshiClient


class MarketSync:
    """Syncs Trump Mentions markets and extracts terms."""

    def __init__(self, client: Optional[KalshiClient] = None):
        self.client = client or KalshiClient()

    def extract_terms_from_market(self, market_data: dict) -> list[dict]:
        """Extract tracked terms from a market title/subtitle.

        Handles formats like:
        - "Will Trump say 'tariff'?" -> ['tariff']
        - "Who are you with / Where are you from" -> compound term
        - "Will Trump mention 'China' or 'trade'?" -> ['china', 'trade']
        """
        title = market_data.get('title', '')
        subtitle = market_data.get('subtitle', '')
        full_text = f"{title} {subtitle}"

        terms = []

        # Pattern 1: Quoted terms - 'term' or "term"
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", full_text)
        for q in quoted:
            q = q.strip()
            if len(q) > 0 and len(q) < 200:
                terms.append(self._build_term_dict(q))

        # Pattern 2: Slash-separated compound terms (e.g., "X / Y")
        if '/' in full_text:
            # Look for patterns like "phrase1 / phrase2"
            slash_parts = re.findall(
                r"['\"]?([^'\"/?]+?)\s*/\s*([^'\"?.]+?)['\"]?(?:\?|$|\s)",
                full_text
            )
            for parts in slash_parts:
                combined = ' / '.join(p.strip() for p in parts if p.strip())
                if combined and len(combined) < 500:
                    sub_terms = [p.strip().lower() for p in parts if p.strip()]
                    terms.append({
                        'term': combined,
                        'normalized_term': combined.lower().strip(),
                        'is_compound': True,
                        'sub_terms': sub_terms,
                    })

        # Pattern 3: Keywords after "say", "mention", "use the word"
        keyword_patterns = [
            r"(?:say|mention|use)\s+(?:the\s+(?:word|phrase|term)\s+)?['\"]?(\w[\w\s]{0,50})['\"]?",
        ]
        for pat in keyword_patterns:
            matches = re.findall(pat, full_text, re.IGNORECASE)
            for m in matches:
                m = m.strip().rstrip('?.,!')
                if m and len(m) < 200 and m.lower() not in [t['normalized_term'] for t in terms]:
                    terms.append(self._build_term_dict(m))

        if not terms:
            # Fallback: use the market title itself as context
            logger.debug(f"No terms extracted from: {full_text}")

        return terms

    def _build_term_dict(self, raw_term: str) -> dict:
        """Build a term dictionary from a raw string."""
        normalized = raw_term.lower().strip()
        is_compound = '/' in raw_term
        sub_terms = None
        if is_compound:
            sub_terms = [p.strip().lower() for p in raw_term.split('/') if p.strip()]
        return {
            'term': raw_term.strip(),
            'normalized_term': normalized,
            'is_compound': is_compound,
            'sub_terms': sub_terms,
        }

    def sync_markets(self) -> dict:
        """Fetch all Trump Mentions markets and sync to database.

        Returns summary of sync results.
        """
        logger.info("Starting market sync...")
        raw_markets = self.client.find_trump_mentions_markets()

        stats = {'markets_found': len(raw_markets), 'new_markets': 0,
                 'updated_markets': 0, 'new_terms': 0}

        with get_session() as session:
            for market_data in raw_markets:
                ticker = market_data.get('ticker', '')
                if not ticker:
                    continue

                # Upsert market
                market = session.query(Market).filter_by(kalshi_ticker=ticker).first()
                if not market:
                    market = Market(kalshi_ticker=ticker)
                    session.add(market)
                    stats['new_markets'] += 1
                else:
                    stats['updated_markets'] += 1

                market.kalshi_event_ticker = market_data.get('event_ticker', '')
                market.title = market_data.get('title', '')
                market.subtitle = market_data.get('subtitle', '')
                market.market_type = 'trump_mentions'
                market.status = market_data.get('status', '')
                market.yes_price = market_data.get('yes_bid', 0) / 100.0 if market_data.get('yes_bid') else None
                market.no_price = market_data.get('no_bid', 0) / 100.0 if market_data.get('no_bid') else None
                market.volume = market_data.get('volume', 0)
                market.open_interest = market_data.get('open_interest', 0)

                if market_data.get('close_time'):
                    try:
                        market.close_time = datetime.fromisoformat(
                            market_data['close_time'].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        pass

                if market_data.get('expiration_time'):
                    try:
                        market.expiration_time = datetime.fromisoformat(
                            market_data['expiration_time'].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        pass

                market.result = market_data.get('result', '')
                market.raw_data = market_data

                # Extract and link terms
                extracted_terms = self.extract_terms_from_market(market_data)
                for term_data in extracted_terms:
                    term = session.query(Term).filter_by(
                        normalized_term=term_data['normalized_term']
                    ).first()
                    if not term:
                        term = Term(
                            term=term_data['term'],
                            normalized_term=term_data['normalized_term'],
                            is_compound=term_data.get('is_compound', False),
                            sub_terms=term_data.get('sub_terms'),
                        )
                        session.add(term)
                        stats['new_terms'] += 1

                    if term not in market.terms:
                        market.terms.append(term)

                session.flush()

        logger.info(f"Market sync complete: {stats}")
        return stats

    def get_all_terms(self) -> list[dict]:
        """Get all tracked terms from the database."""
        with get_session() as session:
            terms = session.query(Term).all()
            return [
                {
                    'id': t.id,
                    'term': t.term,
                    'normalized_term': t.normalized_term,
                    'is_compound': t.is_compound,
                    'sub_terms': t.sub_terms,
                    'total_occurrences': t.total_occurrences,
                    'trend_score': t.trend_score,
                    'market_count': len(t.markets),
                }
                for t in terms
            ]

    def get_active_markets(self) -> list[dict]:
        """Get all active Trump Mentions markets."""
        with get_session() as session:
            markets = session.query(Market).filter(
                Market.status.in_(['active', 'open'])
            ).all()
            return [
                {
                    'id': m.id,
                    'ticker': m.kalshi_ticker,
                    'title': m.title,
                    'subtitle': m.subtitle,
                    'yes_price': m.yes_price,
                    'no_price': m.no_price,
                    'volume': m.volume,
                    'close_time': m.close_time.isoformat() if m.close_time else None,
                    'terms': [t.term for t in m.terms],
                }
                for m in markets
            ]
