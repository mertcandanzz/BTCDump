"""Signal history tracker - records past signals and evaluates outcomes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """A single recorded signal."""
    id: str
    symbol: str
    interval: str
    direction: str
    confidence: float
    price_at_signal: float
    predicted_price: float
    predicted_change_pct: float
    rsi: float
    timestamp: str
    # Outcome (filled later)
    price_after_1h: Optional[float] = None
    price_after_4h: Optional[float] = None
    price_after_24h: Optional[float] = None
    outcome: Optional[str] = None  # "correct" | "wrong" | "pending"


class SignalHistory:
    """Tracks signal history and evaluates outcomes."""

    MAX_RECORDS = 500

    def __init__(self, data_dir: Path = Path("data")) -> None:
        self._file = data_dir / "signal_history.json"
        self._records: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                self._records = json.loads(self._file.read_text())
            except Exception:
                self._records = []

    def _save(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(self._records[-self.MAX_RECORDS:], indent=1))
        except Exception:
            logger.exception("Failed to save signal history")

    def record(self, signal_data: Dict) -> Dict:
        """Record a new signal. Returns the record."""
        ts = datetime.now().isoformat()
        rec = {
            "id": f"{signal_data.get('symbol', 'UNK')}-{int(datetime.now().timestamp())}",
            "symbol": signal_data.get("symbol", ""),
            "interval": signal_data.get("interval", "1h"),
            "direction": signal_data.get("direction", ""),
            "confidence": signal_data.get("confidence", 0),
            "price_at_signal": signal_data.get("current_price", 0),
            "predicted_price": signal_data.get("predicted_price", 0),
            "predicted_change_pct": signal_data.get("change_pct", 0),
            "rsi": signal_data.get("rsi", 0),
            "timestamp": ts,
            "outcome": "pending",
        }
        self._records.append(rec)
        self._save()
        return rec

    def update_outcomes(self, current_prices: Dict[str, float]) -> int:
        """Update outcomes for pending records based on current prices."""
        updated = 0
        now = datetime.now()

        for rec in self._records:
            if rec.get("outcome") != "pending":
                continue

            sym = rec.get("symbol", "")
            price_now = current_prices.get(sym)
            if not price_now:
                continue

            sig_time = datetime.fromisoformat(rec["timestamp"])
            hours_elapsed = (now - sig_time).total_seconds() / 3600
            price_at = rec.get("price_at_signal", 0)

            if not price_at:
                continue

            # Fill in prices at milestones
            if hours_elapsed >= 1 and not rec.get("price_after_1h"):
                rec["price_after_1h"] = price_now
            if hours_elapsed >= 4 and not rec.get("price_after_4h"):
                rec["price_after_4h"] = price_now
            if hours_elapsed >= 24 and not rec.get("price_after_24h"):
                rec["price_after_24h"] = price_now

                # Determine outcome after 24h
                actual_change = (price_now - price_at) / price_at * 100
                direction = rec.get("direction", "")
                if "BUY" in direction:
                    rec["outcome"] = "correct" if actual_change > 0 else "wrong"
                elif "SELL" in direction:
                    rec["outcome"] = "correct" if actual_change < 0 else "wrong"
                else:
                    # HOLD - correct if price didn't move much
                    rec["outcome"] = "correct" if abs(actual_change) < 1 else "wrong"
                updated += 1

        if updated:
            self._save()
        return updated

    def get_history(self, symbol: str = "", limit: int = 50) -> List[Dict]:
        """Get recent signal history."""
        records = self._records
        if symbol:
            records = [r for r in records if r.get("symbol") == symbol]
        return list(reversed(records[-limit:]))

    def get_stats(self, symbol: str = "") -> Dict:
        """Get accuracy statistics."""
        records = self._records
        if symbol:
            records = [r for r in records if r.get("symbol") == symbol]

        total = len(records)
        resolved = [r for r in records if r.get("outcome") in ("correct", "wrong")]
        correct = sum(1 for r in resolved if r["outcome"] == "correct")
        pending = sum(1 for r in records if r.get("outcome") == "pending")

        # By direction
        by_dir = {}
        for r in resolved:
            d = r.get("direction", "UNKNOWN")
            if d not in by_dir:
                by_dir[d] = {"total": 0, "correct": 0}
            by_dir[d]["total"] += 1
            if r["outcome"] == "correct":
                by_dir[d]["correct"] += 1

        return {
            "total_signals": total,
            "resolved": len(resolved),
            "correct": correct,
            "wrong": len(resolved) - correct,
            "pending": pending,
            "accuracy": round(correct / len(resolved) * 100, 1) if resolved else 0,
            "by_direction": {
                k: {**v, "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0}
                for k, v in by_dir.items()
            },
        }
