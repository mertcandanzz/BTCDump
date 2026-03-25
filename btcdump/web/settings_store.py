"""Persistent settings storage - saves/loads platform configuration to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "watchlist": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
    "interval": "1h",
    "notifications": {
        "telegram_token": "",
        "telegram_chat_id": "",
        "discord_webhook": "",
        "enabled": False,
    },
    "webhook_url": "",
    "theme": "dark",
    "auto_refresh_seconds": 0,
    "paper_trading_balance": 10000,
    "trading_rules": [],
    "smart_alerts": [],
    "custom_presets": {},
}


class SettingsStore:
    """File-based settings persistence."""

    def __init__(self, path: Path = Path("data/settings.json")) -> None:
        self._path = path
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict:
        """Load settings from disk."""
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
                logger.info("Settings loaded from %s", self._path)
            except Exception:
                self._data = dict(DEFAULT_SETTINGS)
        else:
            self._data = dict(DEFAULT_SETTINGS)
        return self._data

    def save(self) -> None:
        """Save settings to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))
        except Exception:
            logger.exception("Failed to save settings")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def update(self, data: Dict) -> None:
        self._data.update(data)
        self.save()

    def get_all(self) -> Dict:
        return dict(self._data)
