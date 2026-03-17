"""Email notification system for the trading bot.

Sends trade alerts, daily digests, and critical notifications via SMTP.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from ..config import config
from ..database.db import get_session
from ..database.models import Trade, Market, Term, TermPrediction, Speech


class EmailNotifier:
    """Sends email notifications for trading activity and system status."""

    def __init__(self):
        self.enabled = config.validate_email()
        if not self.enabled:
            logger.info("Email notifications disabled (missing config)")

    def _send_email(self, subject: str, html_body: str) -> bool:
        """Send an email via SMTP."""
        if not self.enabled:
            return False

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config.email_from
        msg['To'] = config.email_to
        msg.attach(MIMEText(html_body, 'html'))

        try:
            with smtplib.SMTP(config.email_smtp_host, config.email_smtp_port) as server:
                server.starttls()
                server.login(config.email_from, config.email_app_password)
                server.sendmail(config.email_from, config.email_to, msg.as_string())
            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_trade_alert(self, trade_details: dict) -> bool:
        """Send immediate alert when a trade is executed or signal detected.

        Handles two data shapes:
        - Full trade execution: has ticker, side, quantity, price_cents, edge, status
        - Trade signal/arbitrage: may only have ticker, side, edge, term, type, etc.
        """
        side = trade_details.get('side', '?').upper()
        ticker = trade_details.get('ticker') or trade_details.get('market_ticker', '?')
        qty = trade_details.get('quantity')
        price = trade_details.get('price_cents')
        edge = trade_details.get('edge', 0)
        status = trade_details.get('status')
        paper = ' [PAPER]' if trade_details.get('paper_mode') else ''

        # Detect if this is a signal-only alert (missing execution details)
        is_signal = qty is None and price is None and status is None
        alert_type = trade_details.get('type', 'Trade')
        term = trade_details.get('term', '')

        if is_signal:
            # Signal / arbitrage alert — show what we have
            heading = f"Trade Signal: {term or alert_type}" if term else f"Signal: {alert_type}"
            subject = f"TrumpGPT Signal: {side} {ticker}"

            rows = f"""
                <tr><td style="padding: 4px 8px; font-weight: bold;">Market</td>
                    <td style="padding: 4px 8px;">{ticker}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Side</td>
                    <td style="padding: 4px 8px;">{side}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Edge</td>
                    <td style="padding: 4px 8px;">{edge:+.1%}</td></tr>"""
            if term:
                rows += f"""
                <tr><td style="padding: 4px 8px; font-weight: bold;">Term</td>
                    <td style="padding: 4px 8px;">{term}</td></tr>"""
            # Include any extra context (e.g. arbitrage profit, action)
            for key in ('action', 'guaranteed_profit', 'reasoning'):
                val = trade_details.get(key)
                if val is not None:
                    label = key.replace('_', ' ').title()
                    display = f"${val:.4f}" if key == 'guaranteed_profit' else val
                    rows += f"""
                <tr><td style="padding: 4px 8px; font-weight: bold;">{label}</td>
                    <td style="padding: 4px 8px;">{display}</td></tr>"""
        else:
            # Full trade execution alert
            heading = f"Trade Executed{paper}"
            subject = f"TrumpGPT{paper}: {side} {qty or 0}x {ticker}"

            rows = f"""
                <tr><td style="padding: 4px 8px; font-weight: bold;">Market</td>
                    <td style="padding: 4px 8px;">{ticker}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Side</td>
                    <td style="padding: 4px 8px;">{side}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Quantity</td>
                    <td style="padding: 4px 8px;">{qty} contracts</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Price</td>
                    <td style="padding: 4px 8px;">{price}c</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Edge</td>
                    <td style="padding: 4px 8px;">{edge:+.1%}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Status</td>
                    <td style="padding: 4px 8px;">{status}</td></tr>"""

        html = f"""
        <div style="font-family: monospace; max-width: 500px;">
            <h2 style="color: {'#2ecc71' if side == 'YES' else '#e74c3c'};">
                {heading}
            </h2>
            <table style="border-collapse: collapse; width: 100%;">
                {rows}
            </table>
            <p style="color: #888; font-size: 12px;">
                {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
            </p>
        </div>
        """
        return self._send_email(subject, html)

    def send_daily_digest(self) -> bool:
        """Send daily summary of bot activity and portfolio status."""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)

        with get_session() as session:
            # Today's trades
            today_trades = session.query(Trade).filter(
                Trade.created_at >= today_start
            ).all()

            # Yesterday's trades (for comparison)
            yesterday_trades = session.query(Trade).filter(
                Trade.created_at >= yesterday_start,
                Trade.created_at < today_start
            ).all()

            # P&L
            daily_pnl = sum(t.pnl or 0 for t in today_trades)
            total_pnl = sum(
                t.pnl or 0 for t in session.query(Trade).filter(
                    Trade.pnl.isnot(None)
                ).all()
            )

            # Active markets
            active_markets = session.query(Market).filter(
                Market.status.in_(['active', 'open'])
            ).count()

            # Recent speeches scraped
            recent_speeches = session.query(Speech).filter(
                Speech.created_at >= yesterday_start
            ).count()

            # Top predictions
            top_preds = session.query(TermPrediction, Term).join(Term).filter(
                TermPrediction.created_at >= yesterday_start
            ).order_by(TermPrediction.probability.desc()).limit(5).all()

            # Build top predictions table
            pred_rows = ""
            for pred, term in top_preds:
                pred_rows += f"""
                <tr>
                    <td style="padding: 4px 8px;">{term.term}</td>
                    <td style="padding: 4px 8px;">{pred.probability:.1%}</td>
                    <td style="padding: 4px 8px;">{pred.confidence:.1%}</td>
                </tr>"""

            # Build trades table
            trade_rows = ""
            for trade in today_trades:
                market = session.query(Market).filter_by(id=trade.market_id).first()
                ticker = market.kalshi_ticker if market else '?'
                pnl_str = f"${trade.pnl:+.2f}" if trade.pnl else "pending"
                trade_rows += f"""
                <tr>
                    <td style="padding: 4px 8px;">{ticker}</td>
                    <td style="padding: 4px 8px;">{trade.side.upper()}</td>
                    <td style="padding: 4px 8px;">{trade.quantity}</td>
                    <td style="padding: 4px 8px;">{pnl_str}</td>
                </tr>"""

        subject = f"TrumpGPT Daily Digest | P&L: ${daily_pnl:+.2f}"

        html = f"""
        <div style="font-family: monospace; max-width: 600px;">
            <h2>TrumpGPT Daily Digest</h2>
            <p style="color: #888;">{now.strftime('%A, %B %d, %Y')}</p>

            <h3>Portfolio</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td style="padding: 4px 8px; font-weight: bold;">Daily P&L</td>
                    <td style="padding: 4px 8px; color: {'#2ecc71' if daily_pnl >= 0 else '#e74c3c'};">
                        ${daily_pnl:+.2f}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">All-Time P&L</td>
                    <td style="padding: 4px 8px; color: {'#2ecc71' if total_pnl >= 0 else '#e74c3c'};">
                        ${total_pnl:+.2f}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Trades Today</td>
                    <td style="padding: 4px 8px;">{len(today_trades)}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Active Markets</td>
                    <td style="padding: 4px 8px;">{active_markets}</td></tr>
                <tr><td style="padding: 4px 8px; font-weight: bold;">Speeches Scraped (24h)</td>
                    <td style="padding: 4px 8px;">{recent_speeches}</td></tr>
            </table>

            {"<h3>Trades Today</h3><table style='border-collapse: collapse; width: 100%;'><tr><th style='padding: 4px 8px; text-align: left;'>Market</th><th style='padding: 4px 8px; text-align: left;'>Side</th><th style='padding: 4px 8px; text-align: left;'>Qty</th><th style='padding: 4px 8px; text-align: left;'>P&L</th></tr>" + trade_rows + "</table>" if trade_rows else "<p>No trades today.</p>"}

            {'<h3>Top Predictions</h3><table style="border-collapse: collapse; width: 100%;"><tr><th style="padding: 4px 8px; text-align: left;">Term</th><th style="padding: 4px 8px; text-align: left;">Prob</th><th style="padding: 4px 8px; text-align: left;">Conf</th></tr>' + pred_rows + '</table>' if pred_rows else ''}

            <hr style="margin-top: 20px;">
            <p style="color: #888; font-size: 11px;">
                TrumpGPT Trading Bot | Running on Raspberry Pi<br>
                {now.strftime('%Y-%m-%d %H:%M UTC')}
            </p>
        </div>
        """
        return self._send_email(subject, html)

    def send_critical_alert(self, title: str, message: str,
                            details: Optional[dict] = None) -> bool:
        """Send critical alert — loss limits, service failures, etc."""
        detail_rows = ""
        if details:
            for k, v in details.items():
                detail_rows += f"""
                <tr><td style="padding: 4px 8px; font-weight: bold;">{k}</td>
                    <td style="padding: 4px 8px;">{v}</td></tr>"""

        html = f"""
        <div style="font-family: monospace; max-width: 500px;">
            <h2 style="color: #e74c3c;">CRITICAL: {title}</h2>
            <p>{message}</p>
            {'<table style="border-collapse: collapse; width: 100%;">' + detail_rows + '</table>' if detail_rows else ''}
            <p style="color: #888; font-size: 12px;">
                {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
            </p>
        </div>
        """
        return self._send_email(f"TrumpGPT CRITICAL: {title}", html)


# Global instance
email_notifier = EmailNotifier()
