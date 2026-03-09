"""Autonomous trading bot for Kalshi Trump Mentions markets."""

import os
from datetime import datetime
from typing import Optional
from loguru import logger

from ..database.models import Trade, Market, BotConfig
from ..database.db import get_session
from .client import KalshiClient
from ..ml.predictor import TermPredictor


class TradingBot:
    """Autonomous trading bot with configurable strategies and risk management."""

    def __init__(self, client: Optional[KalshiClient] = None,
                 predictor: Optional[TermPredictor] = None):
        self.client = client or KalshiClient()
        self.predictor = predictor or TermPredictor()
        self.is_running = False
        self.paper_mode = True  # When True, no real trades are placed

        # Risk parameters
        self.max_position_size = 100  # max contracts per market
        self.max_daily_loss = 50.00  # dollars
        self.max_total_exposure = 500.00  # dollars
        self.min_edge_threshold = 0.05  # minimum edge to trade
        self.min_confidence = 0.3  # minimum prediction confidence
        self.use_kelly = True
        self.kelly_fraction = 0.5  # half-Kelly for safety
        self.auto_trade = False  # manual approval by default

    def get_config(self) -> dict:
        """Get current bot configuration."""
        return {
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss,
            'max_total_exposure': self.max_total_exposure,
            'min_edge_threshold': self.min_edge_threshold,
            'min_confidence': self.min_confidence,
            'use_kelly': self.use_kelly,
            'kelly_fraction': self.kelly_fraction,
            'auto_trade': self.auto_trade,
            'is_running': self.is_running,
        }

    def update_config(self, **kwargs):
        """Update bot configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Bot config updated: {key} = {value}")

    def generate_suggestions(self) -> list[dict]:
        """Generate trading suggestions based on current predictions vs market prices."""
        suggestions = self.predictor.get_trading_suggestions(
            min_edge=self.min_edge_threshold
        )

        # Filter by confidence
        suggestions = [s for s in suggestions if s.get('confidence', 0) >= self.min_confidence]

        # Apply risk management
        for s in suggestions:
            s['suggested_quantity'] = self._calculate_position_size(s)
            s['expected_value'] = round(
                s['edge'] * s['suggested_quantity'] * 1.0,  # $1 per contract
                2
            )

        return suggestions

    def _calculate_position_size(self, suggestion: dict) -> int:
        """Calculate position size based on Kelly criterion and risk limits."""
        if self.use_kelly:
            kelly = suggestion.get('kelly_fraction', 0)
            # Scale Kelly by configured fraction
            adjusted_kelly = kelly * self.kelly_fraction

            # Get balance
            try:
                balance_data = self.client.get_balance()
                balance = balance_data.get('balance', 10000) / 100  # cents to dollars
            except Exception:
                balance = 100  # conservative fallback

            # Kelly-suggested amount
            kelly_dollars = balance * adjusted_kelly
            kelly_contracts = int(kelly_dollars)
        else:
            kelly_contracts = 10  # default

        # Apply limits
        position = min(kelly_contracts, self.max_position_size)
        position = max(1, position)  # at least 1

        # Check total exposure
        current_exposure = self._get_current_exposure()
        remaining = self.max_total_exposure - current_exposure
        if remaining <= 0:
            return 0

        price = suggestion.get('market_yes_price', 0.5)
        cost_per_contract = price if suggestion['suggested_side'] == 'yes' else (1 - price)
        max_by_exposure = int(remaining / max(0.01, cost_per_contract))

        return min(position, max_by_exposure)

    def _get_current_exposure(self) -> float:
        """Calculate current total exposure from open positions."""
        try:
            positions = self.client.get_positions()
            total = 0
            for pos in positions.get('market_positions', []):
                qty = abs(pos.get('position', 0))
                # Rough exposure estimate
                total += qty * 0.50  # assume ~50 cent average price
            return total
        except Exception:
            return 0

    def execute_trade(self, suggestion: dict, quantity: Optional[int] = None,
                      require_confirmation: bool = True) -> Optional[dict]:
        """Execute a trade based on a suggestion.

        Args:
            suggestion: Trading suggestion dict
            quantity: Override quantity (uses suggestion's suggested_quantity if None)
            require_confirmation: If True, returns the order details without placing
        """
        qty = quantity or suggestion.get('suggested_quantity', 1)
        if qty <= 0:
            logger.warning("Position size is 0, skipping trade")
            return None

        ticker = suggestion['market_ticker']
        side = suggestion['suggested_side']

        # Calculate price (in cents for Kalshi API)
        if side == 'yes':
            price_cents = int(suggestion['market_yes_price'] * 100)
        else:
            price_cents = int((1 - suggestion['market_yes_price']) * 100)

        order_details = {
            'ticker': ticker,
            'side': side,
            'action': 'buy',
            'quantity': qty,
            'price_cents': price_cents,
            'edge': suggestion['edge'],
            'reasoning': suggestion.get('reasoning', ''),
        }

        if require_confirmation and not self.auto_trade:
            order_details['status'] = 'pending_confirmation'
            return order_details

        # Paper mode: log but don't place real orders
        if self.paper_mode:
            logger.info(f"[PAPER] Would place: {order_details['action']} {qty}x {side} on {ticker} @ {price_cents}c")
            return {**order_details, 'status': 'paper_trade', 'paper_mode': True}

        # Place the order
        try:
            result = self.client.place_order(
                ticker=ticker,
                side=side,
                action='buy',
                count=qty,
                type='limit',
                yes_price=price_cents if side == 'yes' else None,
                no_price=price_cents if side == 'no' else None,
            )

            # Record the trade
            with get_session() as session:
                market = session.query(Market).filter_by(kalshi_ticker=ticker).first()
                if market:
                    trade = Trade(
                        market_id=market.id,
                        kalshi_order_id=result.get('order', {}).get('order_id', ''),
                        side=side,
                        action='buy',
                        quantity=qty,
                        price=price_cents / 100,
                        status='pending',
                        strategy='ml_predictor',
                        reasoning=suggestion.get('reasoning', ''),
                    )
                    session.add(trade)

            logger.info(f"Order placed: {side} {qty}x {ticker} @ {price_cents}c")
            return {**order_details, 'status': 'placed', 'result': result}

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return {**order_details, 'status': 'error', 'error': str(e)}

    def get_portfolio_summary(self) -> dict:
        """Get summary of current portfolio and performance."""
        try:
            balance = self.client.get_balance()
            positions = self.client.get_positions()
            orders = self.client.get_orders()
        except Exception as e:
            logger.warning(f"Could not fetch portfolio: {e}")
            balance = {}
            positions = {}
            orders = {}

        with get_session() as session:
            total_trades = session.query(Trade).count()
            filled_trades = session.query(Trade).filter_by(status='filled').all()
            total_pnl = sum(t.pnl or 0 for t in filled_trades)

        return {
            'balance': balance.get('balance', 0) / 100 if balance.get('balance') else 0,
            'positions': positions.get('market_positions', []),
            'open_orders': len(orders.get('orders', [])),
            'total_trades': total_trades,
            'total_pnl': total_pnl,
        }

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached."""
        with get_session() as session:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
            today_trades = session.query(Trade).filter(
                Trade.created_at >= today_start,
                Trade.pnl.isnot(None)
            ).all()

            daily_pnl = sum(t.pnl for t in today_trades)
            if daily_pnl < -self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: ${daily_pnl:.2f}")
                return True
            return False
