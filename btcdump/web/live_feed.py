"""Binance WebSocket live price feed."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Dict, List, Set

logger = logging.getLogger(__name__)


class BinanceLiveFeed:
    """Streams real-time price data from Binance WebSocket."""

    WS_URL = "wss://stream.binance.com:9443/stream"

    def __init__(self) -> None:
        self._callbacks: Dict[str, Set[Callable]] = {}
        self._task: asyncio.Task | None = None
        self._symbols: List[str] = []

    async def start(self, symbols: List[str]) -> None:
        """Start streaming for given symbols."""
        self._symbols = [s.upper() for s in symbols]
        for s in self._symbols:
            self._callbacks.setdefault(s, set())
        if self._task:
            self._task.cancel()
        self._task = asyncio.create_task(self._run())

    async def update_symbols(self, symbols: List[str]) -> None:
        """Update symbol list and restart stream."""
        new_syms = [s.upper() for s in symbols]
        if set(new_syms) != set(self._symbols):
            await self.start(new_syms)

    def subscribe(self, symbol: str, callback: Callable) -> None:
        self._callbacks.setdefault(symbol.upper(), set()).add(callback)

    def unsubscribe_all(self, callback: Callable) -> None:
        for cbs in self._callbacks.values():
            cbs.discard(callback)

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed, live feed disabled")
            return

        streams = [f"{s.lower()}@miniTicker" for s in self._symbols if s]
        if not streams:
            return

        url = f"{self.WS_URL}?streams={'/'.join(streams)}"

        while True:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Binance WS connected (%d streams)", len(streams))
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            d = msg.get("data", {})
                            symbol = d.get("s", "")
                            if not symbol or symbol not in self._callbacks:
                                continue

                            open_price = float(d.get("o", 0))
                            close_price = float(d.get("c", 0))
                            tick = {
                                "symbol": symbol,
                                "price": close_price,
                                "change_pct": round(
                                    ((close_price - open_price) / open_price) * 100, 2
                                ) if open_price else 0,
                                "high": float(d.get("h", 0)),
                                "low": float(d.get("l", 0)),
                                "volume": float(d.get("v", 0)),
                            }

                            for cb in list(self._callbacks.get(symbol, [])):
                                try:
                                    await cb(tick)
                                except Exception:
                                    pass
                        except Exception:
                            continue
            except asyncio.CancelledError:
                logger.info("Binance WS feed cancelled")
                return
            except Exception as e:
                logger.warning("Binance WS reconnecting in 5s: %s", e)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
