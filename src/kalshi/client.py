"""Kalshi API client for Trump Mentions markets."""

import os
import time
import requests
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class KalshiClient:
    """Client for interacting with Kalshi's trading API v2."""

    def __init__(self):
        self.base_url = os.getenv('KALSHI_BASE_URL', 'https://trading-api.kalshi.com/trade-api/v2')
        self.email = os.getenv('KALSHI_EMAIL', '')
        self.password = os.getenv('KALSHI_PASSWORD', '')
        self.api_key = os.getenv('KALSHI_API_KEY', '')
        self.token = None
        self.member_id = None
        self._session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 0.1  # rate limiting

    def _rate_limit(self):
        """Simple rate limiter."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _headers(self) -> dict:
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers

    def login(self) -> bool:
        """Authenticate with Kalshi and obtain a session token."""
        try:
            resp = self._session.post(
                f'{self.base_url}/login',
                json={'email': self.email, 'password': self.password},
                headers={'Content-Type': 'application/json'}
            )
            resp.raise_for_status()
            data = resp.json()
            self.token = data.get('token')
            self.member_id = data.get('member_id')
            logger.info(f"Logged in to Kalshi as member {self.member_id}")
            return True
        except Exception as e:
            logger.error(f"Kalshi login failed: {e}")
            return False

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make authenticated GET request."""
        self._rate_limit()
        resp = self._session.get(
            f'{self.base_url}{endpoint}',
            headers=self._headers(),
            params=params or {}
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, data: Optional[dict] = None) -> dict:
        """Make authenticated POST request."""
        self._rate_limit()
        resp = self._session.post(
            f'{self.base_url}{endpoint}',
            headers=self._headers(),
            json=data or {}
        )
        resp.raise_for_status()
        return resp.json()

    def _delete(self, endpoint: str) -> dict:
        """Make authenticated DELETE request."""
        self._rate_limit()
        resp = self._session.delete(
            f'{self.base_url}{endpoint}',
            headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    # --- Market Discovery ---

    def get_events(self, series_ticker: Optional[str] = None,
                   status: Optional[str] = None,
                   cursor: Optional[str] = None,
                   limit: int = 100) -> dict:
        """Fetch events, optionally filtered by series ticker."""
        params = {'limit': limit}
        if series_ticker:
            params['series_ticker'] = series_ticker
        if status:
            params['status'] = status
        if cursor:
            params['cursor'] = cursor
        return self._get('/events', params)

    def get_event(self, event_ticker: str) -> dict:
        """Get a specific event by ticker."""
        return self._get(f'/events/{event_ticker}')

    def get_markets(self, event_ticker: Optional[str] = None,
                    series_ticker: Optional[str] = None,
                    status: Optional[str] = None,
                    cursor: Optional[str] = None,
                    limit: int = 100) -> dict:
        """Fetch markets, with optional filters."""
        params = {'limit': limit}
        if event_ticker:
            params['event_ticker'] = event_ticker
        if series_ticker:
            params['series_ticker'] = series_ticker
        if status:
            params['status'] = status
        if cursor:
            params['cursor'] = cursor
        return self._get('/markets', params)

    def get_market(self, ticker: str) -> dict:
        """Get a specific market by ticker."""
        return self._get(f'/markets/{ticker}')

    def get_market_orderbook(self, ticker: str) -> dict:
        """Get the order book for a market."""
        return self._get(f'/markets/{ticker}/orderbook')

    def get_market_history(self, ticker: str, limit: int = 100,
                           cursor: Optional[str] = None) -> dict:
        """Get trade history for a market."""
        params = {'limit': limit}
        if cursor:
            params['cursor'] = cursor
        return self._get(f'/markets/{ticker}/history', params)

    # --- Trump Mentions specific ---

    def find_trump_mentions_markets(self) -> list[dict]:
        """Find all Trump Mentions markets by searching known series tickers
        and keywords."""
        all_markets = []
        search_terms = [
            'TRUMPMENTIONS', 'TRUMPSAY', 'TRUMP', 'KXTRUMPMENTIONS',
            'POTUSSAY', 'TRUMPWORD',
        ]

        # Search by series tickers
        for term in search_terms:
            try:
                resp = self.get_events(series_ticker=term)
                events = resp.get('events', [])
                for event in events:
                    event_ticker = event.get('event_ticker', '')
                    market_resp = self.get_markets(event_ticker=event_ticker)
                    markets = market_resp.get('markets', [])
                    all_markets.extend(markets)
            except Exception as e:
                logger.debug(f"No results for series {term}: {e}")

        # Also do a broader search
        try:
            cursor = None
            while True:
                resp = self.get_markets(cursor=cursor, limit=200)
                markets = resp.get('markets', [])
                for m in markets:
                    title = (m.get('title', '') + ' ' + m.get('subtitle', '')).lower()
                    if any(kw in title for kw in ['trump', 'say', 'mention', 'word']):
                        if m not in all_markets:
                            all_markets.append(m)
                cursor = resp.get('cursor')
                if not cursor or not markets:
                    break
        except Exception as e:
            logger.warning(f"Broad market search error: {e}")

        # Deduplicate by ticker
        seen = set()
        unique = []
        for m in all_markets:
            ticker = m.get('ticker', '')
            if ticker not in seen:
                seen.add(ticker)
                unique.append(m)

        logger.info(f"Found {len(unique)} Trump Mentions markets")
        return unique

    # --- Trading ---

    def get_balance(self) -> dict:
        """Get account balance."""
        return self._get(f'/portfolio/balance')

    def get_positions(self) -> dict:
        """Get current positions."""
        return self._get('/portfolio/positions')

    def get_orders(self, status: Optional[str] = None) -> dict:
        """Get orders."""
        params = {}
        if status:
            params['status'] = status
        return self._get('/portfolio/orders', params)

    def place_order(self, ticker: str, side: str, action: str,
                    count: int, type: str = 'limit',
                    yes_price: Optional[int] = None,
                    no_price: Optional[int] = None,
                    expiration_ts: Optional[int] = None) -> dict:
        """Place an order on a market.

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            count: Number of contracts
            type: 'limit' or 'market'
            yes_price: Price in cents (1-99) for yes side
            no_price: Price in cents (1-99) for no side
            expiration_ts: Unix timestamp for order expiration
        """
        order = {
            'ticker': ticker,
            'side': side,
            'action': action,
            'count': count,
            'type': type,
        }
        if yes_price is not None:
            order['yes_price'] = yes_price
        if no_price is not None:
            order['no_price'] = no_price
        if expiration_ts is not None:
            order['expiration_ts'] = expiration_ts

        logger.info(f"Placing order: {action} {count}x {side} on {ticker}")
        return self._post('/portfolio/orders', order)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order."""
        return self._delete(f'/portfolio/orders/{order_id}')
