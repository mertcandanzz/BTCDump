"""Price and signal alert system."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    id: str
    symbol: str
    condition: str  # price_above, price_below, signal_buy, signal_sell
    value: float
    triggered: bool = False
    created_at: str = ""
    triggered_at: str = ""

    def check(self, price: float, direction: str) -> bool:
        if self.triggered:
            return False
        if self.condition == "price_above" and price >= self.value:
            return True
        if self.condition == "price_below" and price <= self.value:
            return True
        if self.condition == "signal_buy" and "BUY" in direction:
            return True
        if self.condition == "signal_sell" and "SELL" in direction:
            return True
        return False


class AlertManager:
    def __init__(self) -> None:
        self.alerts: List[Alert] = []

    def add(self, symbol: str, condition: str, value: float) -> Alert:
        alert = Alert(
            id=str(uuid.uuid4())[:8],
            symbol=symbol.upper(),
            condition=condition,
            value=value,
            created_at=datetime.now().isoformat(),
        )
        self.alerts.append(alert)
        logger.info("Alert added: %s %s %s", symbol, condition, value)
        return alert

    def remove(self, alert_id: str) -> bool:
        before = len(self.alerts)
        self.alerts = [a for a in self.alerts if a.id != alert_id]
        return len(self.alerts) < before

    def check(self, symbol: str, price: float, direction: str = "") -> List[Alert]:
        triggered = []
        for alert in self.alerts:
            if alert.symbol != symbol.upper():
                continue
            if alert.check(price, direction):
                alert.triggered = True
                alert.triggered_at = datetime.now().isoformat()
                triggered.append(alert)
                logger.info("Alert triggered: %s %s %s", alert.symbol, alert.condition, alert.value)
        return triggered

    def get_all(self) -> List[Dict]:
        return [
            {"id": a.id, "symbol": a.symbol, "condition": a.condition,
             "value": a.value, "triggered": a.triggered,
             "created_at": a.created_at, "triggered_at": a.triggered_at}
            for a in self.alerts
        ]

    def get_active(self) -> List[Dict]:
        return [d for d in self.get_all() if not d["triggered"]]
