"""Weekly portfolio manager for Kalshi Trump Mentions markets.

Manages the weekly allocation cycle:
1. Sunday night: generate suggested trades for the coming week
2. User approves/rejects pending trades via dashboard
3. Approved trades get executed (paper or live)
4. At week end, contracts settle and P&L is recorded

Auto-approve is OFF by default on every boot — the user must explicitly enable it.
"""

import math
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from ..database.db import get_session
from ..database.models import (
    WeeklyAllocation, PendingTrade, Trade, Market, Term,
    market_term_association, TermPrediction,
)
from .client import KalshiClient
from ..ml.predictor import TermPredictor


class WeeklyPortfolioManager:
    """Manages weekly portfolio allocation cycles.

    Key design:
    - All positions treated as weekly contracts
    - YES-only: we only buy YES (conviction Trump will say the term)
    - Manual approval by default (auto_approve=False on every boot)
    - Full bankroll allocated across highest-conviction terms
    - Paper mode supported via TradingBot
    """

    KALSHI_FEE_ROUND_TRIP = 0.04

    def __init__(self, client: Optional[KalshiClient] = None,
                 predictor: Optional[TermPredictor] = None):
        self.client = client or KalshiClient()
        self.predictor = predictor or TermPredictor()
        self.auto_approve = False  # ALWAYS off on boot
        self.paper_mode = True
        self.max_positions = 15       # max different markets to spread across
        self.min_edge = 0.06          # minimum edge to suggest
        self.min_confidence = 0.4     # minimum prediction confidence
        self.min_probability = 0.3    # minimum predicted probability
        self.kelly_fraction = 0.20    # fraction of Kelly to use

    def get_config(self) -> dict:
        return {
            'auto_approve': self.auto_approve,
            'paper_mode': self.paper_mode,
            'max_positions': self.max_positions,
            'min_edge': self.min_edge,
            'min_confidence': self.min_confidence,
            'kelly_fraction': self.kelly_fraction,
        }

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Portfolio config updated: {key} = {value}")

    # --- Week boundaries ---

    @staticmethod
    def get_current_week_bounds() -> tuple[datetime, datetime]:
        """Get Monday 00:00 UTC to Sunday 23:59 UTC for current week."""
        now = datetime.utcnow()
        monday = now - timedelta(days=now.weekday())
        monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        sunday = monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return monday, sunday

    @staticmethod
    def get_next_week_bounds() -> tuple[datetime, datetime]:
        """Get next week's Monday-Sunday bounds."""
        now = datetime.utcnow()
        days_until_monday = 7 - now.weekday()
        next_monday = now + timedelta(days=days_until_monday)
        next_monday = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        next_sunday = next_monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return next_monday, next_sunday

    # --- Bankroll ---

    def get_bankroll(self) -> float:
        """Get current bankroll from Kalshi or paper balance."""
        if self.paper_mode:
            # Paper mode: start at $100, add settled P&L from past weeks
            with get_session() as session:
                settled = session.query(WeeklyAllocation).filter_by(
                    status='settled'
                ).all()
                base = 100.0
                for alloc in settled:
                    base += (alloc.settled_pnl or 0)
                return max(10.0, base)
        try:
            balance_data = self.client.get_balance()
            return balance_data.get('balance', 10000) / 100
        except Exception:
            return 100.0

    # --- Allocation generation ---

    def generate_weekly_allocation(self) -> Optional[dict]:
        """Generate a new weekly allocation with pending trades.

        Called Sunday night for the coming week, or manually from dashboard.
        Returns the allocation summary or None if already exists.
        """
        week_start, week_end = self.get_next_week_bounds()

        # Check for existing allocation for this week
        with get_session() as session:
            existing = session.query(WeeklyAllocation).filter(
                WeeklyAllocation.week_start == week_start,
            ).first()
            if existing:
                logger.info(f"Allocation already exists for week {week_start.date()}")
                return self._allocation_to_dict(existing, session)

        bankroll = self.get_bankroll()
        logger.info(f"Generating weekly allocation: bankroll=${bankroll:.2f}")

        # Get predictions and market data
        suggestions = self._build_suggestions(bankroll)

        if not suggestions:
            logger.warning("No viable suggestions for weekly allocation")
            return None

        # Create allocation and pending trades
        with get_session() as session:
            allocation = WeeklyAllocation(
                week_start=week_start,
                week_end=week_end,
                bankroll=bankroll,
                allocated_amount=sum(s['total_cost'] for s in suggestions),
                status='pending',
            )
            session.add(allocation)
            session.flush()  # get ID

            for s in suggestions:
                pt = PendingTrade(
                    allocation_id=allocation.id,
                    market_id=s['market_id'],
                    term=s['term'],
                    side='yes',
                    suggested_quantity=s['quantity'],
                    suggested_price=s['price'],
                    total_cost=s['total_cost'],
                    our_probability=s['probability'],
                    edge=s['edge'],
                    confidence=s['confidence'],
                    recency_score=s.get('recency_score'),
                    kelly_fraction=s.get('kelly_fraction', 0),
                    reasoning=s.get('reasoning', ''),
                    status='pending',
                )
                session.add(pt)

            result = self._allocation_to_dict(allocation, session)

        logger.info(
            f"Generated allocation: {len(suggestions)} trades, "
            f"${sum(s['total_cost'] for s in suggestions):.2f} allocated"
        )
        return result

    def generate_current_week_allocation(self) -> Optional[dict]:
        """Generate allocation for the current week (for manual trigger)."""
        week_start, week_end = self.get_current_week_bounds()

        with get_session() as session:
            existing = session.query(WeeklyAllocation).filter(
                WeeklyAllocation.week_start == week_start,
            ).first()
            if existing:
                return self._allocation_to_dict(existing, session)

        bankroll = self.get_bankroll()
        suggestions = self._build_suggestions(bankroll)

        if not suggestions:
            return None

        with get_session() as session:
            allocation = WeeklyAllocation(
                week_start=week_start,
                week_end=week_end,
                bankroll=bankroll,
                allocated_amount=sum(s['total_cost'] for s in suggestions),
                status='pending',
            )
            session.add(allocation)
            session.flush()

            for s in suggestions:
                pt = PendingTrade(
                    allocation_id=allocation.id,
                    market_id=s['market_id'],
                    term=s['term'],
                    side='yes',
                    suggested_quantity=s['quantity'],
                    suggested_price=s['price'],
                    total_cost=s['total_cost'],
                    our_probability=s['probability'],
                    edge=s['edge'],
                    confidence=s['confidence'],
                    recency_score=s.get('recency_score'),
                    kelly_fraction=s.get('kelly_fraction', 0),
                    reasoning=s.get('reasoning', ''),
                    status='pending',
                )
                session.add(pt)

            return self._allocation_to_dict(allocation, session)

    def _build_suggestions(self, bankroll: float) -> list[dict]:
        """Build ranked list of trade suggestions within bankroll."""
        suggestions_raw = self.predictor.get_trading_suggestions(
            min_edge=self.min_edge
        )

        # Filter: YES only, confidence, probability
        filtered = []
        for s in suggestions_raw:
            if s['suggested_side'] != 'yes':
                continue
            if s.get('confidence', 0) < self.min_confidence:
                continue
            if s.get('our_probability', 0) < self.min_probability:
                continue
            # Net edge after fees must be positive
            if abs(s['edge']) - self.KALSHI_FEE_ROUND_TRIP <= 0:
                continue
            # Skip 1c/99c markets
            price = s.get('market_yes_price', 0.5)
            if price <= 0.01 or price >= 0.99:
                continue
            filtered.append(s)

        # Sort by edge * confidence (best opportunities first)
        filtered.sort(
            key=lambda x: abs(x['edge']) * x.get('confidence', 0.5),
            reverse=True,
        )

        # Take top N positions
        filtered = filtered[:self.max_positions]

        # Allocate bankroll using Kelly sizing
        suggestions = []
        remaining = bankroll

        for s in filtered:
            if remaining <= 1.0:
                break

            # Kelly fraction sizing
            kelly = s.get('kelly_fraction', 0)
            adjusted = kelly * self.kelly_fraction
            confidence = s.get('confidence', 0.5)
            adjusted *= confidence

            dollars = bankroll * adjusted
            dollars = min(dollars, remaining)
            dollars = max(1.0, dollars)

            price = s['market_yes_price']
            quantity = max(1, int(dollars / max(0.01, price)))
            actual_cost = quantity * price

            if actual_cost > remaining:
                quantity = max(1, int(remaining / max(0.01, price)))
                actual_cost = quantity * price

            if quantity <= 0:
                continue

            # Get market_id
            market_id = None
            with get_session() as session:
                market = session.query(Market).filter_by(
                    kalshi_ticker=s['market_ticker']
                ).first()
                if market:
                    market_id = market.id

            if not market_id:
                continue

            # Get recency score if available
            recency_score = None
            try:
                pred_data = s.get('component_scores', {})
                if pred_data:
                    recency_score = pred_data.get('market_recency')
            except Exception:
                pass

            suggestions.append({
                'market_id': market_id,
                'ticker': s['market_ticker'],
                'term': s['term'],
                'quantity': quantity,
                'price': price,
                'total_cost': round(actual_cost, 2),
                'probability': s['our_probability'],
                'edge': s['edge'],
                'confidence': s.get('confidence', 0.5),
                'recency_score': recency_score,
                'kelly_fraction': kelly,
                'reasoning': s.get('reasoning', ''),
            })

            remaining -= actual_cost

        return suggestions

    # --- Approval flow ---

    def approve_trade(self, pending_trade_id: int) -> Optional[dict]:
        """Approve a pending trade for execution."""
        with get_session() as session:
            pt = session.query(PendingTrade).filter_by(id=pending_trade_id).first()
            if not pt:
                return None
            if pt.status != 'pending':
                return {'error': f'Trade already {pt.status}'}

            pt.status = 'approved'
            pt.approved_at = datetime.utcnow()

            # Mark allocation as active
            alloc = pt.allocation
            if alloc and alloc.status == 'pending':
                alloc.status = 'active'

            return self._pending_trade_to_dict(pt, session)

    def reject_trade(self, pending_trade_id: int) -> Optional[dict]:
        """Reject a pending trade."""
        with get_session() as session:
            pt = session.query(PendingTrade).filter_by(id=pending_trade_id).first()
            if not pt:
                return None
            if pt.status != 'pending':
                return {'error': f'Trade already {pt.status}'}

            pt.status = 'rejected'
            return self._pending_trade_to_dict(pt, session)

    def approve_all_pending(self, allocation_id: Optional[int] = None) -> int:
        """Approve all pending trades (for auto-approve or bulk approval)."""
        count = 0
        with get_session() as session:
            query = session.query(PendingTrade).filter_by(status='pending')
            if allocation_id:
                query = query.filter_by(allocation_id=allocation_id)
            pending = query.all()

            for pt in pending:
                pt.status = 'approved'
                pt.approved_at = datetime.utcnow()
                count += 1

                if pt.allocation and pt.allocation.status == 'pending':
                    pt.allocation.status = 'active'

        logger.info(f"Bulk approved {count} pending trades")
        return count

    def reject_all_pending(self, allocation_id: Optional[int] = None) -> int:
        """Reject all pending trades."""
        count = 0
        with get_session() as session:
            query = session.query(PendingTrade).filter_by(status='pending')
            if allocation_id:
                query = query.filter_by(allocation_id=allocation_id)
            for pt in query.all():
                pt.status = 'rejected'
                count += 1
        return count

    # --- Execution ---

    def execute_approved_trades(self) -> list[dict]:
        """Execute all approved pending trades.

        Called periodically by scheduler or after approval.
        """
        executed = []

        with get_session() as session:
            approved = session.query(PendingTrade).filter_by(
                status='approved'
            ).all()

            for pt in approved:
                market = session.query(Market).filter_by(id=pt.market_id).first()
                if not market:
                    pt.status = 'expired'
                    continue

                # Check market is still active
                if market.status not in ('active', 'open'):
                    pt.status = 'expired'
                    continue

                price_cents = int(pt.suggested_price * 100)

                if self.paper_mode:
                    # Paper trade
                    trade = Trade(
                        market_id=pt.market_id,
                        side='yes',
                        action='buy',
                        quantity=pt.suggested_quantity,
                        price=pt.suggested_price,
                        fill_price=pt.suggested_price,
                        status='paper',
                        strategy='weekly_portfolio',
                        reasoning=pt.reasoning,
                    )
                    session.add(trade)
                    session.flush()

                    pt.status = 'executed'
                    pt.executed_at = datetime.utcnow()
                    pt.execution_price = pt.suggested_price
                    pt.trade_id = trade.id

                    executed.append({
                        'ticker': market.kalshi_ticker,
                        'term': pt.term,
                        'qty': pt.suggested_quantity,
                        'price': pt.suggested_price,
                        'status': 'paper_trade',
                    })
                    logger.info(
                        f"[PAPER] Weekly: YES {pt.suggested_quantity}x "
                        f"{market.kalshi_ticker} @ {price_cents}c"
                    )
                else:
                    # Live trade
                    try:
                        result = self.client.place_order(
                            ticker=market.kalshi_ticker,
                            side='yes',
                            action='buy',
                            count=pt.suggested_quantity,
                            type='limit',
                            yes_price=price_cents,
                        )

                        trade = Trade(
                            market_id=pt.market_id,
                            kalshi_order_id=result.get('order', {}).get('order_id', ''),
                            side='yes',
                            action='buy',
                            quantity=pt.suggested_quantity,
                            price=pt.suggested_price,
                            status='pending',
                            strategy='weekly_portfolio',
                            reasoning=pt.reasoning,
                        )
                        session.add(trade)
                        session.flush()

                        pt.status = 'executed'
                        pt.executed_at = datetime.utcnow()
                        pt.execution_price = pt.suggested_price
                        pt.trade_id = trade.id

                        executed.append({
                            'ticker': market.kalshi_ticker,
                            'term': pt.term,
                            'qty': pt.suggested_quantity,
                            'price': pt.suggested_price,
                            'status': 'placed',
                        })
                    except Exception as e:
                        logger.error(f"Weekly trade failed for {market.kalshi_ticker}: {e}")

        if executed:
            logger.info(f"Executed {len(executed)} weekly trades")
        return executed

    # --- Settlement ---

    def check_settlements(self) -> list[dict]:
        """Check for settled markets and update allocation P&L.

        For paper mode: check if market.result is set.
        For live mode: also check Kalshi API for settled positions.
        """
        settled_results = []

        with get_session() as session:
            # Find executed trades in active allocations
            active_allocs = session.query(WeeklyAllocation).filter(
                WeeklyAllocation.status.in_(['active', 'pending'])
            ).all()

            for alloc in active_allocs:
                alloc_pnl = 0.0
                all_settled = True

                for pt in alloc.pending_trades:
                    if pt.status != 'executed':
                        if pt.status == 'pending':
                            all_settled = False
                        continue

                    market = session.query(Market).filter_by(id=pt.market_id).first()
                    if not market:
                        continue

                    if market.result in ('yes', 'no'):
                        # Market settled
                        cost = pt.suggested_quantity * (pt.execution_price or pt.suggested_price)
                        if market.result == 'yes':
                            # YES settled: we get $1 per contract
                            revenue = pt.suggested_quantity * 1.0
                            pnl = revenue - cost
                        else:
                            # NO settled: we lose our cost
                            pnl = -cost

                        # Subtract fees
                        pnl -= pt.suggested_quantity * self.KALSHI_FEE_ROUND_TRIP / 2
                        alloc_pnl += pnl

                        settled_results.append({
                            'term': pt.term,
                            'ticker': market.kalshi_ticker,
                            'result': market.result,
                            'quantity': pt.suggested_quantity,
                            'entry_price': pt.execution_price or pt.suggested_price,
                            'pnl': round(pnl, 2),
                        })

                        # Update the linked trade P&L
                        if pt.trade_id:
                            trade = session.query(Trade).filter_by(id=pt.trade_id).first()
                            if trade and trade.pnl is None:
                                trade.pnl = round(pnl, 2)
                                trade.status = 'filled'
                    else:
                        all_settled = False

                # If week is past and all executed trades settled
                if all_settled and alloc.week_end < datetime.utcnow():
                    alloc.settled_pnl = round(alloc_pnl, 2)
                    alloc.status = 'settled'
                    logger.info(
                        f"Week {alloc.week_start.date()} settled: "
                        f"P&L ${alloc_pnl:+.2f}"
                    )

        return settled_results

    # --- Queries ---

    def get_current_allocation(self) -> Optional[dict]:
        """Get the current week's allocation."""
        week_start, _ = self.get_current_week_bounds()
        with get_session() as session:
            alloc = session.query(WeeklyAllocation).filter(
                WeeklyAllocation.week_start == week_start,
            ).first()
            if alloc:
                return self._allocation_to_dict(alloc, session)
        return None

    def get_pending_trades(self, allocation_id: Optional[int] = None) -> list[dict]:
        """Get all pending trades, optionally filtered by allocation."""
        with get_session() as session:
            query = session.query(PendingTrade)
            if allocation_id:
                query = query.filter_by(allocation_id=allocation_id)
            else:
                query = query.filter_by(status='pending')
            trades = query.order_by(PendingTrade.edge.desc()).all()
            return [self._pending_trade_to_dict(pt, session) for pt in trades]

    def get_all_trades_for_allocation(self, allocation_id: int) -> list[dict]:
        """Get all trades (any status) for a given allocation."""
        with get_session() as session:
            trades = session.query(PendingTrade).filter_by(
                allocation_id=allocation_id
            ).order_by(PendingTrade.created_at).all()
            return [self._pending_trade_to_dict(pt, session) for pt in trades]

    def get_allocation_history(self, limit: int = 12) -> list[dict]:
        """Get past weekly allocation summaries."""
        with get_session() as session:
            allocs = session.query(WeeklyAllocation).order_by(
                WeeklyAllocation.week_start.desc()
            ).limit(limit).all()
            return [self._allocation_to_dict(a, session) for a in allocs]

    def get_positions(self) -> list[dict]:
        """Get current held positions from executed weekly trades."""
        positions = []
        with get_session() as session:
            # Get executed pending trades from active allocations
            active_allocs = session.query(WeeklyAllocation).filter(
                WeeklyAllocation.status.in_(['active', 'pending'])
            ).all()

            for alloc in active_allocs:
                for pt in alloc.pending_trades:
                    if pt.status != 'executed':
                        continue

                    market = session.query(Market).filter_by(id=pt.market_id).first()
                    if not market:
                        continue

                    current_price = market.yes_price or 0.5
                    entry = pt.execution_price or pt.suggested_price
                    unrealized = (current_price - entry) * pt.suggested_quantity

                    positions.append({
                        'ticker': market.kalshi_ticker,
                        'title': market.title or market.kalshi_ticker,
                        'term': pt.term,
                        'side': 'yes',
                        'qty': pt.suggested_quantity,
                        'entry_price': round(entry, 4),
                        'current_price': round(current_price, 4),
                        'cost_basis': round(entry * pt.suggested_quantity, 2),
                        'current_value': round(current_price * pt.suggested_quantity, 2),
                        'unrealized_pnl': round(unrealized, 2),
                        'unrealized_pct': round(
                            (current_price - entry) / max(0.01, entry) * 100, 1
                        ),
                        'market_status': market.status,
                        'market_result': market.result,
                        'close_time': market.close_time.isoformat() if market.close_time else None,
                        'week_start': alloc.week_start.isoformat(),
                    })

        positions.sort(key=lambda x: abs(x['unrealized_pnl']), reverse=True)
        return positions

    def get_portfolio_summary(self) -> dict:
        """Get overall portfolio summary across all weeks."""
        with get_session() as session:
            allocs = session.query(WeeklyAllocation).all()

            total_settled_pnl = sum(a.settled_pnl or 0 for a in allocs if a.status == 'settled')
            weeks_traded = len([a for a in allocs if a.status == 'settled'])
            winning_weeks = len([a for a in allocs if a.status == 'settled' and (a.settled_pnl or 0) > 0])

            # Current week
            current = None
            week_start, _ = self.get_current_week_bounds()
            for a in allocs:
                if a.week_start == week_start:
                    current = a
                    break

            current_positions = self.get_positions()
            unrealized = sum(p['unrealized_pnl'] for p in current_positions)

            return {
                'bankroll': round(self.get_bankroll(), 2),
                'total_settled_pnl': round(total_settled_pnl, 2),
                'unrealized_pnl': round(unrealized, 2),
                'total_pnl': round(total_settled_pnl + unrealized, 2),
                'weeks_traded': weeks_traded,
                'winning_weeks': winning_weeks,
                'win_rate': round(winning_weeks / max(1, weeks_traded) * 100, 1),
                'positions_count': len(current_positions),
                'current_week_status': current.status if current else 'none',
                'current_week_allocated': round(current.allocated_amount, 2) if current else 0,
                'pending_trades': self._count_pending(),
                'paper_mode': self.paper_mode,
                'auto_approve': self.auto_approve,
            }

    def _count_pending(self) -> int:
        with get_session() as session:
            return session.query(PendingTrade).filter_by(status='pending').count()

    # --- Serialization helpers ---

    def _allocation_to_dict(self, alloc: WeeklyAllocation, session) -> dict:
        trades = [self._pending_trade_to_dict(pt, session) for pt in alloc.pending_trades]
        pending_count = sum(1 for t in trades if t['status'] == 'pending')
        approved_count = sum(1 for t in trades if t['status'] in ('approved', 'executed'))
        rejected_count = sum(1 for t in trades if t['status'] == 'rejected')

        return {
            'id': alloc.id,
            'week_start': alloc.week_start.isoformat(),
            'week_end': alloc.week_end.isoformat(),
            'bankroll': round(alloc.bankroll, 2),
            'allocated_amount': round(alloc.allocated_amount, 2),
            'settled_pnl': round(alloc.settled_pnl, 2) if alloc.settled_pnl is not None else None,
            'status': alloc.status,
            'trades': trades,
            'trade_count': len(trades),
            'pending_count': pending_count,
            'approved_count': approved_count,
            'rejected_count': rejected_count,
            'created_at': alloc.created_at.isoformat() if alloc.created_at else None,
        }

    def _pending_trade_to_dict(self, pt: PendingTrade, session) -> dict:
        market = session.query(Market).filter_by(id=pt.market_id).first()
        ticker = market.kalshi_ticker if market else 'unknown'
        current_price = market.yes_price if market else pt.suggested_price
        market_status = market.status if market else 'unknown'
        close_time = market.close_time.isoformat() if market and market.close_time else None

        return {
            'id': pt.id,
            'allocation_id': pt.allocation_id,
            'ticker': ticker,
            'term': pt.term,
            'side': pt.side,
            'quantity': pt.suggested_quantity,
            'suggested_price': round(pt.suggested_price, 4),
            'current_price': round(current_price, 4) if current_price else None,
            'total_cost': round(pt.total_cost, 2),
            'probability': round(pt.our_probability, 4) if pt.our_probability else None,
            'edge': round(pt.edge, 4) if pt.edge else None,
            'confidence': round(pt.confidence, 4) if pt.confidence else None,
            'recency_score': round(pt.recency_score, 4) if pt.recency_score else None,
            'kelly_fraction': round(pt.kelly_fraction, 4) if pt.kelly_fraction else None,
            'status': pt.status,
            'approved_at': pt.approved_at.isoformat() if pt.approved_at else None,
            'executed_at': pt.executed_at.isoformat() if pt.executed_at else None,
            'execution_price': round(pt.execution_price, 4) if pt.execution_price else None,
            'reasoning': pt.reasoning,
            'market_status': market_status,
            'close_time': close_time,
            'created_at': pt.created_at.isoformat() if pt.created_at else None,
        }
