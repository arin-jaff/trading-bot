"""Kalshi API client for Trump Mentions markets.

Uses RSA key-pair authentication for trading endpoints.
Public endpoints (markets, events) work without auth.
"""

import os
import time
import base64
import requests
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class KalshiClient:
    """Client for interacting with Kalshi's trading API v2.

    Authentication uses RSA key signing:
    - KALSHI_API_KEY: Your API key ID
    - KALSHI_PRIVATE_KEY_PATH: Path to your RSA private key (.pem file)

    Public endpoints (markets, events) work without authentication.
    Trading endpoints (portfolio, orders) require auth.
    """

    BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'

    def __init__(self):
        self.base_url = os.getenv('KALSHI_BASE_URL', self.BASE_URL)
        self.api_key_id = os.getenv('KALSHI_API_KEY', '')
        self.private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH', '')
        self._auth = None
        self._session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 0.2  # rate limiting (Kalshi has strict limits)
        self._authenticated = False

    def _rate_limit(self):
        """Simple rate limiter."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _sign_request(self, method: str, path: str) -> dict:
        """Create RSA-PSS signed auth headers for a request."""
        if not self._auth:
            return {'Content-Type': 'application/json'}

        try:
            timestamp_ms = str(int(time.time() * 1000))
            # Message to sign: timestamp + METHOD + path
            msg = (timestamp_ms + method.upper() + path).encode('utf-8')

            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            signature = self._auth['private_key'].sign(
                msg,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )

            return {
                'Content-Type': 'application/json',
                'KALSHI-ACCESS-KEY': self._auth['key_id'],
                'KALSHI-ACCESS-SIGNATURE': base64.b64encode(signature).decode('utf-8'),
                'KALSHI-ACCESS-TIMESTAMP': timestamp_ms,
            }
        except Exception as e:
            logger.error(f"Request signing failed: {e}")
            return {'Content-Type': 'application/json'}

    def login(self) -> bool:
        """Set up RSA key authentication.

        Loads the private key and verifies it works by calling the balance endpoint.
        """
        if not self.api_key_id:
            logger.warning("No KALSHI_API_KEY configured. Trading features disabled. "
                          "Market data still works without auth.")
            return False

        if not self.private_key_path or not os.path.exists(self.private_key_path):
            logger.warning(f"RSA private key not found at '{self.private_key_path}'. "
                          "Generate one at kalshi.com/account/api. "
                          "Set KALSHI_PRIVATE_KEY_PATH in .env")
            return False

        try:
            from cryptography.hazmat.primitives import serialization

            with open(self.private_key_path, 'rb') as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)

            self._auth = {
                'key_id': self.api_key_id,
                'private_key': private_key,
            }

            # Verify auth works
            balance = self.get_balance()
            self._authenticated = True
            logger.info(f"Kalshi auth OK. Balance: ${balance.get('balance', 0) / 100:.2f}")
            return True

        except Exception as e:
            logger.error(f"Kalshi auth failed: {e}")
            self._auth = None
            return False

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    def _full_path(self, endpoint: str) -> str:
        """Get the full path for signing (must include /trade-api/v2 prefix)."""
        return f'/trade-api/v2{endpoint}'

    def _get(self, endpoint: str, params: Optional[dict] = None,
             require_auth: bool = False) -> dict:
        """Make a GET request. Auth headers added if available."""
        self._rate_limit()
        url = f'{self.base_url}{endpoint}'
        full_path = self._full_path(endpoint)
        headers = self._sign_request('GET', full_path) if self._auth else {'Content-Type': 'application/json'}

        resp = self._session.get(url, headers=headers, params=params or {})
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, data: Optional[dict] = None) -> dict:
        """Make authenticated POST request."""
        self._rate_limit()
        url = f'{self.base_url}{endpoint}'
        full_path = self._full_path(endpoint)
        headers = self._sign_request('POST', full_path)

        resp = self._session.post(url, headers=headers, json=data or {})
        resp.raise_for_status()
        return resp.json()

    def _delete(self, endpoint: str) -> dict:
        """Make authenticated DELETE request."""
        self._rate_limit()
        url = f'{self.base_url}{endpoint}'
        full_path = self._full_path(endpoint)
        headers = self._sign_request('DELETE', full_path)

        resp = self._session.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # --- Market Discovery (public, no auth needed) ---

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
        """Find all Trump Mentions markets.

        Uses the known series ticker KXTRUMPSAY and fetches all events/markets.
        """
        all_markets = []

        # Primary: KXTRUMPSAY is the confirmed series ticker
        series_tickers = ['KXTRUMPSAY']

        for series in series_tickers:
            try:
                cursor = None
                while True:
                    resp = self.get_events(series_ticker=series, cursor=cursor)
                    events = resp.get('events', [])
                    if not events:
                        break

                    for event in events:
                        event_ticker = event.get('event_ticker', '')
                        # Fetch all markets for this event
                        market_cursor = None
                        while True:
                            market_resp = self.get_markets(
                                event_ticker=event_ticker, cursor=market_cursor
                            )
                            markets = market_resp.get('markets', [])
                            all_markets.extend(markets)
                            market_cursor = market_resp.get('cursor')
                            if not market_cursor or not markets:
                                break

                    cursor = resp.get('cursor')
                    if not cursor:
                        break

            except Exception as e:
                logger.error(f"Error fetching series {series}: {e}")

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

    # --- Trading (requires auth) ---

    def get_balance(self) -> dict:
        """Get account balance."""
        return self._get('/portfolio/balance')

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
