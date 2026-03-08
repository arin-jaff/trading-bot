"""Alert system for live events, trading opportunities, and term detections."""

import os
import json
import subprocess
from datetime import datetime
from typing import Optional
from loguru import logger


class AlertManager:
    """Manages alerts and notifications for the trading bot."""

    def __init__(self):
        self.alerts: list[dict] = []
        self.max_alerts = 500

    def add_alert(self, alert_type: str, title: str, message: str,
                  severity: str = 'info', data: Optional[dict] = None):
        """Add a new alert.

        Args:
            alert_type: 'live_event', 'trade_signal', 'term_detection', 'system'
            title: Short alert title
            message: Alert message
            severity: 'info', 'warning', 'critical'
            data: Additional structured data
        """
        alert = {
            'id': len(self.alerts),
            'type': alert_type,
            'title': title,
            'message': message,
            'severity': severity,
            'data': data,
            'timestamp': datetime.utcnow().isoformat(),
            'read': False,
        }

        self.alerts.append(alert)

        # Trim old alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        # Desktop notification for critical alerts
        if severity == 'critical':
            self._desktop_notification(title, message)

        logger.info(f"[ALERT][{severity}] {title}: {message}")

    def _desktop_notification(self, title: str, message: str):
        """Send a macOS desktop notification."""
        try:
            subprocess.run([
                'osascript', '-e',
                f'display notification "{message}" with title "Trump Bot: {title}" sound name "Glass"'
            ], timeout=5, capture_output=True)
        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")

    def get_recent_alerts(self, limit: int = 50,
                          alert_type: Optional[str] = None,
                          unread_only: bool = False) -> list[dict]:
        """Get recent alerts, optionally filtered."""
        alerts = self.alerts.copy()

        if alert_type:
            alerts = [a for a in alerts if a['type'] == alert_type]
        if unread_only:
            alerts = [a for a in alerts if not a['read']]

        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]

    def mark_read(self, alert_id: int):
        """Mark an alert as read."""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['read'] = True
                break

    def get_unread_count(self) -> int:
        return sum(1 for a in self.alerts if not a['read'])

    # --- Convenience methods ---

    def alert_event_live(self, event_title: str, event_type: str):
        self.add_alert(
            'live_event',
            'Trump Speaking LIVE',
            f'{event_type}: {event_title}',
            severity='critical',
            data={'event_title': event_title, 'event_type': event_type},
        )

    def alert_trade_signal(self, term: str, side: str, edge: float,
                           market_ticker: str):
        self.add_alert(
            'trade_signal',
            f'Trade Signal: {term}',
            f'{side.upper()} signal on {market_ticker} (edge: {edge:+.1%})',
            severity='warning',
            data={'term': term, 'side': side, 'edge': edge, 'ticker': market_ticker},
        )

    def alert_term_detected(self, term: str, count: int, source: str):
        self.add_alert(
            'term_detection',
            f'Term Detected: {term}',
            f'"{term}" mentioned {count}x ({source})',
            severity='info',
            data={'term': term, 'count': count, 'source': source},
        )


# Global instance
alert_manager = AlertManager()
