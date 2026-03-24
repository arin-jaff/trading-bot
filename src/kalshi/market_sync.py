"""Sync Kalshi Trump Mentions markets to local database."""

import re
from datetime import datetime
from typing import Optional
from loguru import logger
from sqlalchemy.orm import Session

from ..database.models import Market, Term, PriceSnapshot, market_term_association
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

        # Pattern 0: custom_strike.Word field (Kalshi's actual term field)
        custom_strike = market_data.get('custom_strike', {})
        word = custom_strike.get('Word', '')
        if word:
            terms.append(self._build_term_dict(word))
            # If the Word contains a slash (e.g., "Doge/Dogecoin"), also add as compound
            if '/' in word:
                sub_terms = [p.strip().lower() for p in word.split('/') if p.strip()]
                terms.append({
                    'term': word,
                    'normalized_term': word.lower().strip(),
                    'is_compound': True,
                    'sub_terms': sub_terms,
                })

        # Also check yes_sub_title and no_sub_title
        for field in ['yes_sub_title', 'no_sub_title']:
            val = market_data.get(field, '')
            if val and val.lower() not in [t['normalized_term'] for t in terms]:
                terms.append(self._build_term_dict(val))

        # Pattern 1: Quoted terms - 'term' or "term"
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", full_text)
        for q in quoted:
            q = q.strip()
            if len(q) > 0 and len(q) < 200:
                terms.append(self._build_term_dict(q))

        # Pattern 2: Slash-separated compound terms (e.g., "X / Y")
        # Handled via custom_strike.Word above — the regex below is a
        # fallback for titles only.  The Word field is the authoritative
        # source and already captures the full phrase on both sides of
        # the slash, so we skip this pattern if we already got a
        # compound term from Pattern 0.
        has_compound_from_word = any(t.get('is_compound') for t in terms)
        if '/' in full_text and not has_compound_from_word:
            # Look for patterns like "phrase1 / phrase2"
            slash_parts = re.findall(
                r"['\"]?([^'\"/?]+?)\s*/\s*([^'\"?]+?)['\"]?(?:\?|$)",
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

    @staticmethod
    def _parse_dollar_field(data: dict, *field_names: str) -> float | None:
        """Parse a Kalshi v2 dollar-string field (e.g., '0.9900') to float.

        Tries each field name in order, returns the first valid value.
        """
        for name in field_names:
            val = data.get(name)
            if val is not None:
                try:
                    f = float(val)
                    if f > 0:
                        return f
                except (ValueError, TypeError):
                    continue
        return None

    @staticmethod
    def _parse_fp_field(data: dict, *field_names: str) -> int:
        """Parse a Kalshi v2 fixed-point field (e.g., '37061.00') to int.

        Tries each field name in order, returns the first valid value.
        """
        for name in field_names:
            val = data.get(name)
            if val is not None:
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    continue
        return 0

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

    # How many markets to process per commit batch (reduces lock hold time)
    _BATCH_SIZE = 10

    def sync_markets(self) -> dict:
        """Fetch all Trump Mentions markets and sync to database.

        Returns summary of sync results.
        3A: Detects newly created markets and adds them to new_market_tickers.

        Markets are committed in batches to avoid holding the DB write lock
        for the entire sync (which can take 5-15s with 100+ markets).
        """
        logger.info("Starting market sync...")
        raw_markets = self.client.find_trump_mentions_markets()

        stats = {'markets_found': len(raw_markets), 'new_markets': 0,
                 'updated_markets': 0, 'new_terms': 0,
                 'new_market_tickers': []}  # 3A: track new markets

        # Process in batches to keep write lock hold time short
        for batch_start in range(0, len(raw_markets), self._BATCH_SIZE):
            batch = raw_markets[batch_start:batch_start + self._BATCH_SIZE]
            with get_session() as session:
                for market_data in batch:
                    ticker = market_data.get('ticker', '')
                    if not ticker:
                        continue

                    # Upsert market
                    market = session.query(Market).filter_by(kalshi_ticker=ticker).first()
                    if not market:
                        market = Market(kalshi_ticker=ticker)
                        session.add(market)
                        stats['new_markets'] += 1
                        stats['new_market_tickers'].append(ticker)  # 3A
                    else:
                        stats['updated_markets'] += 1

                    market.kalshi_event_ticker = market_data.get('event_ticker', '')
                    market.title = market_data.get('title', '')
                    market.subtitle = market_data.get('subtitle', '')
                    market.market_type = 'trump_mentions'
                    market.status = market_data.get('status', '')

                    # Prices: v2 API uses dollar-string fields (*_dollars)
                    # Fallback chain: yes_bid_dollars -> last_price_dollars -> legacy yes_bid (cents)
                    yes_price = self._parse_dollar_field(
                        market_data, 'yes_bid_dollars', 'last_price_dollars'
                    )
                    no_price = self._parse_dollar_field(
                        market_data, 'no_bid_dollars'
                    )
                    # Legacy fallback: old API returned cents (0-100)
                    if yes_price is None:
                        legacy = market_data.get('yes_bid') or market_data.get('last_price')
                        if legacy:
                            yes_price = legacy / 100.0
                    if no_price is None:
                        legacy = market_data.get('no_bid')
                        if legacy:
                            no_price = legacy / 100.0

                    market.yes_price = yes_price
                    market.no_price = no_price

                    # Volume: v2 uses volume_fp (fixed-point string), fallback to volume
                    market.volume = self._parse_fp_field(
                        market_data, 'volume_fp', 'volume'
                    )
                    market.open_interest = self._parse_fp_field(
                        market_data, 'open_interest_fp', 'open_interest'
                    )

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

                    # Flush so new markets get an ID before we create price snapshots
                    session.flush()

                    # Record price snapshot for history (must be after flush for new markets)
                    if yes_price is not None:
                        snapshot = PriceSnapshot(
                            market_id=market.id,
                            yes_price=yes_price,
                            no_price=no_price,
                            volume=market.volume,
                            open_interest=market.open_interest,
                        )
                        session.add(snapshot)

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
