"""Paper trading engine - virtual portfolio management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    id: str
    symbol: str
    side: str  # long | short
    entry_price: float
    quantity: float
    entry_time: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def unrealized_pnl(self, price: float) -> float:
        if self.side == "long":
            return (price - self.entry_price) * self.quantity
        return (self.entry_price - price) * self.quantity

    def unrealized_pnl_pct(self, price: float) -> float:
        cost = self.entry_price * self.quantity
        return (self.unrealized_pnl(price) / cost) * 100 if cost else 0


@dataclass
class ClosedTrade:
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float


class PaperTrader:
    def __init__(self, initial_balance: float = 10000.0) -> None:
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.history: List[ClosedTrade] = []
        self.journal: Dict[str, List[Dict]] = {}  # trade_id -> notes

    def open_position(self, symbol: str, side: str, price: float,
                      size_pct: float = 10, stop_loss: float = 0, take_profit: float = 0) -> Dict:
        if symbol in self.positions:
            raise ValueError(f"Already have position in {symbol}")
        trade_value = self.balance * (size_pct / 100)
        if trade_value <= 0:
            raise ValueError("Insufficient balance")
        quantity = trade_value / price
        self.balance -= trade_value
        pos = Position(
            id=f"{symbol}-{int(datetime.now().timestamp())}",
            symbol=symbol, side=side, entry_price=price, quantity=quantity,
            entry_time=datetime.now().isoformat(),
            stop_loss=stop_loss if stop_loss > 0 else None,
            take_profit=take_profit if take_profit > 0 else None,
        )
        self.positions[symbol] = pos
        logger.info("Paper %s %s @ $%.2f qty=%.6f", side, symbol, price, quantity)
        return self._pos_dict(pos, price)

    def close_position(self, symbol: str, price: float) -> Dict:
        pos = self.positions.pop(symbol, None)
        if not pos:
            raise ValueError(f"No position in {symbol}")
        pnl = pos.unrealized_pnl(price)
        pnl_pct = pos.unrealized_pnl_pct(price)
        self.balance += (pos.entry_price * pos.quantity) + pnl
        trade = ClosedTrade(id=pos.id, symbol=symbol, side=pos.side,
                            entry_price=pos.entry_price, exit_price=price,
                            quantity=pos.quantity, entry_time=pos.entry_time,
                            exit_time=datetime.now().isoformat(),
                            pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2))
        self.history.append(trade)
        return {"symbol": symbol, "pnl": trade.pnl, "pnl_pct": trade.pnl_pct}

    def check_sl_tp(self, symbol: str, price: float) -> Optional[Dict]:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        if pos.side == "long":
            if pos.stop_loss and price <= pos.stop_loss:
                return self.close_position(symbol, price)
            if pos.take_profit and price >= pos.take_profit:
                return self.close_position(symbol, price)
        else:
            if pos.stop_loss and price >= pos.stop_loss:
                return self.close_position(symbol, price)
            if pos.take_profit and price <= pos.take_profit:
                return self.close_position(symbol, price)
        return None

    def get_portfolio(self, current_prices: Dict[str, float]) -> Dict:
        total_value = self.balance
        positions = []
        for sym, pos in self.positions.items():
            cp = current_prices.get(sym, pos.entry_price)
            total_value += (pos.entry_price * pos.quantity) + pos.unrealized_pnl(cp)
            positions.append(self._pos_dict(pos, cp))
        return {
            "balance": round(self.balance, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(total_value - self.initial_balance, 2),
            "total_pnl_pct": round((total_value / self.initial_balance - 1) * 100, 2),
            "positions": positions,
            "total_trades": len(self.history),
            "win_rate": round(sum(1 for t in self.history if t.pnl > 0) / len(self.history), 3) if self.history else 0,
        }

    def get_history(self) -> List[Dict]:
        return [{"id": t.id, "symbol": t.symbol, "side": t.side, "entry": t.entry_price,
                 "exit": t.exit_price, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
                 "entry_time": t.entry_time, "exit_time": t.exit_time}
                for t in reversed(self.history)]

    def reset(self):
        self.balance = self.initial_balance
        self.positions.clear()
        self.history.clear()
        self.journal.clear()

    def add_note(self, trade_id: str, note: str) -> Dict:
        """Add a journal note to a trade."""
        entry = {
            "note": note,
            "timestamp": datetime.now().isoformat(),
        }
        if trade_id not in self.journal:
            self.journal[trade_id] = []
        self.journal[trade_id].append(entry)
        return entry

    def get_journal(self, trade_id: str = "") -> Dict:
        """Get journal entries."""
        if trade_id:
            return {"trade_id": trade_id, "notes": self.journal.get(trade_id, [])}
        return {"all_notes": {k: v for k, v in self.journal.items() if v}}

    @staticmethod
    def _pos_dict(pos, price):
        return {"id": pos.id, "symbol": pos.symbol, "side": pos.side,
                "entry_price": pos.entry_price, "current_price": price,
                "quantity": round(pos.quantity, 6),
                "pnl": round(pos.unrealized_pnl(price), 2),
                "pnl_pct": round(pos.unrealized_pnl_pct(price), 2),
                "stop_loss": pos.stop_loss, "take_profit": pos.take_profit,
                "entry_time": pos.entry_time}
