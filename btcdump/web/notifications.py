"""Telegram/Discord notification system for signal changes."""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class NotificationManager:
    """Sends notifications when signals change direction."""

    def __init__(self) -> None:
        self.telegram_token: str = ""
        self.telegram_chat_id: str = ""
        self.discord_webhook: str = ""
        self.enabled: bool = False
        self._previous_signals: Dict[str, str] = {}

    def configure(self, telegram_token="", telegram_chat_id="", discord_webhook="", enabled=True):
        if telegram_token:
            self.telegram_token = telegram_token
        if telegram_chat_id:
            self.telegram_chat_id = telegram_chat_id
        if discord_webhook:
            self.discord_webhook = discord_webhook
        self.enabled = enabled and bool(self.telegram_token or self.discord_webhook)

    async def check_and_notify(self, symbol: str, signal_data: Dict) -> Optional[str]:
        """Send notification if signal direction changed. Returns message if sent."""
        if not self.enabled:
            return None

        current_dir = signal_data.get("direction", "")
        prev_dir = self._previous_signals.get(symbol, "")
        self._previous_signals[symbol] = current_dir

        if not prev_dir or current_dir == prev_dir:
            return None

        msg = self._format(symbol, signal_data, prev_dir)

        if self.telegram_token and self.telegram_chat_id:
            await self._send_telegram(msg)
        if self.discord_webhook:
            await self._send_discord(msg)

        logger.info("Notification sent: %s %s -> %s", symbol, prev_dir, current_dir)
        return msg

    def _format(self, symbol: str, data: Dict, prev_dir: str) -> str:
        emoji = {"STRONG BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡",
                 "SELL": "🔴", "STRONG SELL": "🔴🔴"}.get(data.get("direction", ""), "⚪")
        return (
            f"{emoji} Signal Change: {symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Previous: {prev_dir}\n"
            f"Current:  {data.get('direction','')} (Conf: {data.get('confidence',0):.0f}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Price:      ${data.get('current_price',0):,.2f}\n"
            f"Prediction: ${data.get('predicted_price',0):,.2f} ({data.get('change_pct',0):+.2f}%)\n"
            f"RSI: {data.get('rsi','?')} | R/R: {data.get('risk_reward','?')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"BTCDump AI Signal Engine"
        )

    async def _send_telegram(self, msg: str) -> None:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={"chat_id": self.telegram_chat_id, "text": msg},
                    timeout=10,
                )
        except Exception as e:
            logger.error("Telegram failed: %s", e)

    async def _send_discord(self, msg: str) -> None:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(self.discord_webhook, json={"content": msg}, timeout=10)
        except Exception as e:
            logger.error("Discord failed: %s", e)

    def get_status(self) -> Dict:
        return {
            "enabled": self.enabled,
            "has_telegram": bool(self.telegram_token and self.telegram_chat_id),
            "has_discord": bool(self.discord_webhook),
        }
