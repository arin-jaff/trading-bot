"""Autonomous trading bot for Kalshi Trump Mentions markets.

Position-aware: tracks what we hold and only trades the delta.
YES-only by default: buying YES = genuine conviction Trump will say the term.
Rate-limited: max trades per day, max 1 entry per market per day, cooldowns.
"""

from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from ..database.models import Trade, Market
from ..database.db import get_session
from .client import KalshiClient
from ..ml.predictor import TermPredictor


class TradingBot:
    """Autonomous trading bot with position-aware risk management.

    Key safeguards against over-trading:
    1. Position tracking: checks current holdings before every trade
    2. Per-market cooldown: max 1 new entry per market per day
    3. Global daily cap: max N total new orders per day
    4. YES-only mode: only buys YES contracts (requires genuine edge/conviction)
    5. Delta-based sizing: desired_position - current_position = trade size
    """

    # Kalshi fee: ~2% per side (buy + sell/settle), 4% round-trip
    KALSHI_FEE_PER_SIDE = 0.02
    KALSHI_FEE_ROUND_TRIP = 0.04

    # Trade rate limits
    MAX_TRADES_PER_DAY = 10          # absolute cap on new orders per day
    MAX_TRADES_PER_MARKET_PER_DAY = 1  # max 1 new entry per market per day
    MAX_TRADES_PER_CYCLE = 2         # max new orders per 5-min cycle

    # Max sell orders per manage_positions() cycle
    MAX_SELLS_PER_CYCLE = 5
    SELL_DELAY_SECONDS = 1.0

    def __init__(self, client: Optional[KalshiClient] = None,
                 predictor: Optional[TermPredictor] = None):
        self.client = client or KalshiClient()
        self.predictor = predictor or TermPredictor()
        self.is_running = False
        self.paper_mode = True  # When True, no real trades are placed

        # Risk parameters
        self.max_position_size = 25   # max contracts per market (was 100)
        self.max_daily_loss = 50.00   # dollars
        self.max_total_exposure = 200.00  # dollars (was 500)
        self.min_edge_threshold = 0.08  # minimum edge to trade (was 0.05)
        self.min_confidence = 0.5     # minimum prediction confidence (was 0.3)
        self.min_volume = 50          # minimum market volume
        self.use_kelly = True
        self.kelly_fraction = 0.25    # quarter-Kelly for safety (was 0.5)
        self.auto_trade = True        # paper mode keeps it safe
        self.yes_only = True          # only buy YES contracts

        # Drawdown protection
        self.max_drawdown_pct = 0.30
        self._peak_balance = None
        self._cooldown_until = None

        # In-memory trade tracking (reset daily)
        self._trades_today: dict[str, datetime] = {}  # ticker -> last trade time
        self._daily_trade_count = 0
        self._last_reset_date: Optional[str] = None

    def _reset_daily_counters(self):
        """Reset daily tracking counters at midnight."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if self._last_reset_date != today:
            self._trades_today = {}
            self._daily_trade_count = 0
            self._last_reset_date = today

    def get_config(self) -> dict:
        """Get current bot configuration."""
        self._reset_daily_counters()
        return {
            'paper_mode': self.paper_mode,
            'auto_trade': self.auto_trade,
            'yes_only': self.yes_only,
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss,
            'max_total_exposure': self.max_total_exposure,
            'min_edge_threshold': self.min_edge_threshold,
            'min_confidence': self.min_confidence,
            'min_volume': self.min_volume,
            'use_kelly': self.use_kelly,
            'kelly_fraction': self.kelly_fraction,
            'fee_round_trip': self.KALSHI_FEE_ROUND_TRIP,
            'is_running': self.is_running,
            'trades_today': self._daily_trade_count,
            'max_trades_per_day': self.MAX_TRADES_PER_DAY,
        }

    def update_config(self, **kwargs):
        """Update bot configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Bot config updated: {key} = {value}")

    # --- Position tracking ---

    def _get_held_positions(self) -> dict:
        """Get currently held positions from trade history.

        Returns: {ticker: {side, qty, avg_entry_price, market_id, total_cost}}
        Uses the DB trade log to reconstruct positions (works for both paper and live).
        """
        positions = {}
        with get_session() as session:
            trades = session.query(Trade).join(Market).filter(
                Trade.status.in_(['filled', 'paper', 'pending'])
            ).order_by(Trade.created_at).all()

            for t in trades:
                ticker = t.market.kalshi_ticker if t.market else None
                if not ticker:
                    continue

                if ticker not in positions:
                    positions[ticker] = {
                        'side': t.side, 'qty': 0,
                        'total_cost': 0.0, 'market_id': t.market_id,
                    }

                price = t.fill_price or t.price or 0.5

                if t.action == 'buy':
                    positions[ticker]['qty'] += t.quantity
                    positions[ticker]['total_cost'] += t.quantity * price
                    positions[ticker]['side'] = t.side
                elif t.action == 'sell':
                    positions[ticker]['qty'] = max(
                        0, positions[ticker]['qty'] - t.quantity
                    )
                    positions[ticker]['total_cost'] -= t.quantity * price

        # Remove closed positions
        return {
            k: {
                **v,
                'avg_entry_price': round(
                    v['total_cost'] / v['qty'], 4
                ) if v['qty'] > 0 else 0,
            }
            for k, v in positions.items() if v['qty'] > 0
        }

    def get_positions_detail(self) -> list[dict]:
        """Get detailed position info with current P&L for the UI."""
        held = self._get_held_positions()
        result = []

        with get_session() as session:
            for ticker, pos in held.items():
                market = session.query(Market).filter_by(
                    kalshi_ticker=ticker
                ).first()
                if not market:
                    continue

                current_price = market.yes_price or 0.5
                if pos['side'] == 'no':
                    current_price = 1.0 - current_price

                entry = pos['avg_entry_price']
                unrealized_per = current_price - entry
                unrealized_total = unrealized_per * pos['qty']

                # Calculate cost basis and current value
                cost_basis = entry * pos['qty']
                current_value = current_price * pos['qty']

                # Get term names for display
                terms = [t.term for t in market.terms] if market.terms else []

                result.append({
                    'ticker': ticker,
                    'title': market.title or ticker,
                    'terms': terms,
                    'side': pos['side'],
                    'qty': pos['qty'],
                    'avg_entry': round(entry, 4),
                    'current_price': round(current_price, 4),
                    'cost_basis': round(cost_basis, 2),
                    'current_value': round(current_value, 2),
                    'unrealized_pnl': round(unrealized_total, 2),
                    'unrealized_pct': round(
                        unrealized_per / max(0.01, entry) * 100, 1
                    ),
                    'market_status': market.status,
                    'close_time': market.close_time.isoformat() if market.close_time else None,
                })

        result.sort(key=lambda x: abs(x['unrealized_pnl']), reverse=True)
        return result

    # --- Suggestion generation ---

    def generate_suggestions(self) -> list[dict]:
        """Generate trading suggestions, filtered by position awareness.

        Filters applied:
        1. Confidence >= min_confidence
        2. Net edge after fees > 0
        3. Liquidity (volume >= min_volume)
        4. YES-only mode (if enabled)
        5. Not already at max position for this market
        6. Not already traded this market today
        7. Under daily trade cap
        """
        self._reset_daily_counters()

        suggestions = self.predictor.get_trading_suggestions(
            min_edge=self.min_edge_threshold
        )

        # Filter by confidence
        suggestions = [
            s for s in suggestions
            if s.get('confidence', 0) >= self.min_confidence
        ]

        # Filter: net edge after fees must be positive
        suggestions = [
            s for s in suggestions
            if abs(s['edge']) - self.KALSHI_FEE_ROUND_TRIP > 0
        ]

        # Filter: liquidity
        suggestions = [
            s for s in suggestions
            if s.get('volume', self.min_volume) >= self.min_volume
        ]

        # Filter: YES-only mode — only trade when we think Trump WILL say it
        if self.yes_only:
            suggestions = [
                s for s in suggestions
                if s['suggested_side'] == 'yes'
            ]

        # Get current positions to calculate deltas
        held = self._get_held_positions()

        filtered = []
        for s in suggestions:
            ticker = s['market_ticker']

            # Skip if already traded this market today
            if ticker in self._trades_today:
                continue

            # Calculate desired position vs current
            current_qty = held.get(ticker, {}).get('qty', 0)
            desired_qty = self._calculate_position_size(s)

            # Only trade if we need more contracts
            delta = desired_qty - current_qty
            if delta <= 0:
                continue

            s['suggested_quantity'] = delta
            s['current_position'] = current_qty
            s['desired_position'] = desired_qty

            # Net expected value after fees
            net_edge = abs(s['edge']) - self.KALSHI_FEE_ROUND_TRIP
            s['expected_value'] = round(net_edge * delta, 2)

            filtered.append(s)

        # Sort by expected value (best opportunities first)
        filtered.sort(key=lambda x: x['expected_value'], reverse=True)
        return filtered

    def _calculate_position_size(self, suggestion: dict) -> int:
        """Calculate desired position size using fee-aware Kelly criterion.

        Returns the TOTAL desired position (not the delta to trade).
        """
        if self.use_kelly:
            kelly = suggestion.get('kelly_fraction', 0)
            adjusted_kelly = kelly * self.kelly_fraction

            # Scale by confidence
            confidence = suggestion.get('confidence', 0.5)
            adjusted_kelly *= confidence

            # Get balance
            try:
                balance_data = self.client.get_balance()
                balance = balance_data.get('balance', 10000) / 100
            except Exception:
                balance = 100

            kelly_dollars = balance * adjusted_kelly
            kelly_contracts = int(kelly_dollars)
        else:
            kelly_contracts = 5

        # Time-to-close decay
        close_time = suggestion.get('close_time')
        if close_time:
            try:
                if isinstance(close_time, str):
                    close_dt = datetime.fromisoformat(
                        close_time.replace('Z', '+00:00')
                    )
                else:
                    close_dt = close_time
                hours_left = max(
                    0, (close_dt - datetime.utcnow()).total_seconds() / 3600
                )
                if hours_left < 2:
                    kelly_contracts = int(kelly_contracts * 0.5)
                elif hours_left > 120:
                    kelly_contracts = int(kelly_contracts * 0.3)
            except Exception:
                pass

        # Apply limits
        position = min(kelly_contracts, self.max_position_size)
        position = max(1, position)

        # Cap to 10% of market volume
        volume = suggestion.get('volume', 0)
        if volume > 0:
            max_by_volume = max(1, int(volume * 0.10))
            position = min(position, max_by_volume)

        # Check total exposure
        current_exposure = self._get_current_exposure()
        remaining = self.max_total_exposure - current_exposure
        if remaining <= 0:
            return 0

        price = suggestion.get('market_yes_price', 0.5)
        cost_per_contract = (
            price if suggestion['suggested_side'] == 'yes' else (1 - price)
        )
        max_by_exposure = int(remaining / max(0.01, cost_per_contract))

        return min(position, max_by_exposure)

    def _get_current_exposure(self) -> float:
        """Calculate total exposure from held positions."""
        held = self._get_held_positions()
        total = 0.0
        with get_session() as session:
            for ticker, pos in held.items():
                market = session.query(Market).filter_by(
                    kalshi_ticker=ticker
                ).first()
                price = market.yes_price if market and market.yes_price else 0.50
                cost = price if pos['side'] == 'yes' else (1 - price)
                total += pos['qty'] * cost
        return total

    # --- Trade execution ---

    def execute_trade(self, suggestion: dict, quantity: Optional[int] = None,
                      require_confirmation: bool = True) -> Optional[dict]:
        """Execute a trade with position-awareness and rate limiting."""
        self._reset_daily_counters()

        qty = quantity or suggestion.get('suggested_quantity', 1)
        if qty <= 0:
            return None

        ticker = suggestion['market_ticker']
        side = suggestion['suggested_side']

        # Rate limit checks
        if self._daily_trade_count >= self.MAX_TRADES_PER_DAY:
            logger.info(
                f"Daily trade cap reached ({self.MAX_TRADES_PER_DAY}), "
                f"skipping {ticker}"
            )
            return None

        if ticker in self._trades_today:
            logger.debug(f"Already traded {ticker} today, skipping")
            return None

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
            'edge': suggestion.get('edge', 0),
            'reasoning': suggestion.get('reasoning', ''),
        }

        if require_confirmation and not self.auto_trade:
            order_details['status'] = 'pending_confirmation'
            return order_details

        # Record the trade attempt in rate-limit tracking
        self._trades_today[ticker] = datetime.utcnow()
        self._daily_trade_count += 1

        # Paper mode
        if self.paper_mode:
            logger.info(
                f"[PAPER] {side.upper()} {qty}x {ticker} @ {price_cents}c "
                f"(trade #{self._daily_trade_count}/{self.MAX_TRADES_PER_DAY} today)"
            )
            with get_session() as session:
                market = session.query(Market).filter_by(
                    kalshi_ticker=ticker
                ).first()
                if market:
                    trade = Trade(
                        market_id=market.id,
                        side=side,
                        action='buy',
                        quantity=qty,
                        price=price_cents / 100,
                        status='paper',
                        strategy='ml_predictor',
                        reasoning=suggestion.get('reasoning', ''),
                    )
                    session.add(trade)
            return {**order_details, 'status': 'paper_trade', 'paper_mode': True}

        # Live order
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

            with get_session() as session:
                market = session.query(Market).filter_by(
                    kalshi_ticker=ticker
                ).first()
                if market:
                    trade = Trade(
                        market_id=market.id,
                        kalshi_order_id=result.get('order', {}).get(
                            'order_id', ''
                        ),
                        side=side,
                        action='buy',
                        quantity=qty,
                        price=price_cents / 100,
                        status='pending',
                        strategy='ml_predictor',
                        reasoning=suggestion.get('reasoning', ''),
                    )
                    session.add(trade)

            logger.info(
                f"Order placed: {side} {qty}x {ticker} @ {price_cents}c "
                f"(#{self._daily_trade_count}/{self.MAX_TRADES_PER_DAY})"
            )
            return {**order_details, 'status': 'placed', 'result': result}

        except Exception as e:
            logger.error(f"Order failed: {e}")
            # Undo rate-limit tracking on failure
            self._trades_today.pop(ticker, None)
            self._daily_trade_count = max(0, self._daily_trade_count - 1)
            return {**order_details, 'status': 'error', 'error': str(e)}

    # --- Portfolio ---

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

        held = self._get_held_positions()

        with get_session() as session:
            total_trades = session.query(Trade).count()
            filled_trades = session.query(Trade).filter(
                Trade.status.in_(['filled', 'paper'])
            ).all()
            total_pnl = sum(t.pnl or 0 for t in filled_trades)

        return {
            'balance': (
                balance.get('balance', 0) / 100 if balance.get('balance') else 0
            ),
            'positions': positions.get('market_positions', []),
            'held_count': len(held),
            'held_contracts': sum(p['qty'] for p in held.values()),
            'open_orders': len(orders.get('orders', [])),
            'total_trades': total_trades,
            'total_pnl': total_pnl,
        }

    # --- Risk management ---

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached."""
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            logger.info(
                f"Trading paused until "
                f"{self._cooldown_until.strftime('%H:%M UTC')}"
            )
            return True

        # Also check daily trade cap
        self._reset_daily_counters()
        if self._daily_trade_count >= self.MAX_TRADES_PER_DAY:
            logger.info(
                f"Daily trade cap reached: "
                f"{self._daily_trade_count}/{self.MAX_TRADES_PER_DAY}"
            )
            return True

        with get_session() as session:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
            today_trades = session.query(Trade).filter(
                Trade.created_at >= today_start,
                Trade.pnl.isnot(None)
            ).all()

            daily_pnl = sum(t.pnl for t in today_trades)

            if daily_pnl < -self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: ${daily_pnl:.2f}")
                self._send_loss_alert(daily_pnl, 'daily_limit')
                return True

            if daily_pnl < -(self.max_daily_loss * 0.5) and not self._cooldown_until:
                self._cooldown_until = datetime.utcnow() + timedelta(hours=2)
                logger.warning(
                    f"Entering 2-hour cooldown after ${daily_pnl:.2f} daily loss"
                )
                self._send_loss_alert(daily_pnl, 'cooldown')
                return True

        if self._check_drawdown():
            return True

        return False

    def _check_drawdown(self) -> bool:
        """Halt trading if balance has dropped too far from peak."""
        try:
            balance_data = self.client.get_balance()
            balance = balance_data.get('balance', 0) / 100
        except Exception:
            return False

        if self._peak_balance is None:
            self._peak_balance = balance

        if balance > self._peak_balance:
            self._peak_balance = balance

        if self._peak_balance > 0:
            drawdown = (self._peak_balance - balance) / self._peak_balance
            if drawdown >= self.max_drawdown_pct:
                logger.warning(
                    f"Drawdown protection triggered: {drawdown:.1%} "
                    f"(peak=${self._peak_balance:.2f}, now=${balance:.2f})"
                )
                self._send_loss_alert(
                    balance - self._peak_balance, 'drawdown',
                    details={
                        'peak': self._peak_balance, 'current': balance,
                        'drawdown_pct': f"{drawdown:.1%}",
                    }
                )
                return True
        return False

    def _send_loss_alert(self, pnl: float, reason: str,
                         details: Optional[dict] = None):  # noqa: ARG002
        """Log loss alert."""
        logger.warning(f"Loss alert: {reason}, P&L: ${pnl:+.2f}")

    # --- Position management ---

    def manage_positions(self) -> list[dict]:
        """Active position management: profit-taking and stop-loss.

        Sell orders are rate-limited (1s apart, max 5 per cycle).
        Stop-losses are prioritized over profit-taking.
        """
        actions = []
        try:
            positions = self.client.get_positions()
        except Exception as e:
            logger.debug(f"Could not fetch positions for management: {e}")
            return actions

        for pos in positions.get('market_positions', []):
            ticker = pos.get('ticker', '')
            qty = pos.get('position', 0)
            if qty == 0:
                continue

            with get_session() as session:
                market = session.query(Market).filter_by(
                    kalshi_ticker=ticker
                ).first()
                if not market:
                    continue

                last_trade = session.query(Trade).filter_by(
                    market_id=market.id, status='filled'
                ).order_by(Trade.created_at.desc()).first()

                if not last_trade:
                    continue

                entry_price = last_trade.fill_price or last_trade.price
                current_price = market.yes_price or 0.5
                if last_trade.side == 'no':
                    current_price = 1.0 - current_price

                unrealized = current_price - entry_price

                if unrealized > self.KALSHI_FEE_ROUND_TRIP * 2:
                    sell_qty = max(1, abs(qty) // 2)
                    actions.append({
                        'type': 'take_profit',
                        'ticker': ticker,
                        'side': last_trade.side,
                        'quantity': sell_qty,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'unrealized': round(unrealized, 4),
                    })
                elif unrealized < -0.15:
                    actions.append({
                        'type': 'stop_loss',
                        'ticker': ticker,
                        'side': last_trade.side,
                        'quantity': abs(qty),
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'unrealized': round(unrealized, 4),
                    })

        if actions:
            logger.info(f"Position management: {len(actions)} actions")

        if self.auto_trade:
            import time as _time
            actions.sort(
                key=lambda a: (
                    0 if a['type'] == 'stop_loss' else 1, a['unrealized']
                )
            )
            executed = 0
            for action in actions:
                if executed >= self.MAX_SELLS_PER_CYCLE:
                    logger.warning(
                        f"Hit sell cap ({self.MAX_SELLS_PER_CYCLE}/cycle), "
                        f"deferring {len(actions) - executed} remaining"
                    )
                    break
                if executed > 0:
                    _time.sleep(self.SELL_DELAY_SECONDS)
                self._execute_sell(
                    action['ticker'], action['side'],
                    action['quantity'], action['current_price'],
                )
                executed += 1

        return actions

    def _execute_sell(self, ticker: str, side: str, quantity: int,
                      price: float):
        """Execute a sell order."""
        if self.paper_mode:
            logger.info(
                f"[PAPER] Would sell {quantity}x {side} on {ticker} "
                f"@ {price:.2f}"
            )
            return

        try:
            price_cents = int(price * 100)
            self.client.place_order(
                ticker=ticker,
                side=side,
                action='sell',
                count=quantity,
                type='limit',
                yes_price=price_cents if side == 'yes' else None,
                no_price=price_cents if side == 'no' else None,
            )
            logger.info(
                f"Sold {quantity}x {side} on {ticker} @ {price_cents}c"
            )
        except Exception as e:
            logger.error(f"Sell order failed for {ticker}: {e}")
