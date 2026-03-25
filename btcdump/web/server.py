"""FastAPI server: REST + WebSocket for BTCDump multi-coin Web UI."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from btcdump.config import AppConfig
from btcdump.data import DataFetcher
from btcdump.models import ModelPipeline
from btcdump.signals import SignalGenerator
from btcdump.utils import ensure_dirs, setup_logging
from btcdump.web.coin_manager import CoinManager
from btcdump.web.discussion import DiscussionEngine
from btcdump.web.llm import LLMManager, PROVIDER_MODELS

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


class BTCDumpWebApp:
    """Main web application state."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.fetcher = DataFetcher(self.config.data)
        self.pipeline = ModelPipeline(self.config)
        self.signal_gen = SignalGenerator(self.config.signal)
        self.llm_manager = LLMManager()
        self.discussion = DiscussionEngine(self.llm_manager)

        # Multi-coin manager (replaces old single-coin state)
        self.coin_manager = CoinManager(
            self.fetcher, self.pipeline, self.signal_gen, self.config,
        )

        # Chat histories per LLM provider
        self.chat_histories: Dict[str, list] = {p: [] for p in PROVIDER_MODELS}

        ensure_dirs(self.config.data.cache_dir, self.config.model.models_dir)
        setup_logging(self.config.log_level, self.config.log_file)


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    state = BTCDumpWebApp(config)

    app = FastAPI(title="BTCDump", version="4.0.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # ── HTML ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    # ── Coin Discovery ────────────────────────────────────

    @app.get("/api/coins")
    async def get_coins(q: str = "", limit: int = 100):
        try:
            coins = await asyncio.to_thread(state.coin_manager.get_coins, q, limit)
            return {"ok": True, "coins": coins}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/coin/{symbol}/signal")
    async def get_coin_signal(symbol: str):
        try:
            data = await asyncio.to_thread(state.coin_manager.compute_signal, symbol)
            return {"ok": True, "data": data, "status": data.get("status", "ready")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/coin/{symbol}/chart-data")
    async def get_chart_data(symbol: str):
        try:
            prices = await asyncio.to_thread(
                state.coin_manager.get_mini_chart_data, symbol,
            )
            return {"ok": True, "prices": prices, "symbol": symbol, "count": len(prices)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/coin/{symbol}/ohlcv")
    async def get_ohlcv(symbol: str, limit: int = 100):
        """Full OHLCV data for candlestick chart."""
        try:
            def _fetch():
                interval = state.coin_manager.active_interval
                data = state.coin_manager.fetcher.fetch_with_cache(symbol, interval)
                df = data.df.tail(limit)
                return [{
                    "t": int(r["time"].timestamp() * 1000),
                    "o": round(float(r["open"]), 6),
                    "h": round(float(r["high"]), 6),
                    "l": round(float(r["low"]), 6),
                    "c": round(float(r["close"]), 6),
                    "v": round(float(r["volume"]), 2),
                } for _, r in df.iterrows()]
            candles = await asyncio.to_thread(_fetch)
            return {"ok": True, "candles": candles, "symbol": symbol}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/coin/select")
    async def select_coin(body: dict):
        symbol = body.get("symbol", "").upper()
        state.coin_manager.set_active_coin(symbol)
        return {"ok": True, "symbol": symbol}

    # ── Watchlist ─────────────────────────────────────────

    @app.post("/api/watchlist")
    async def set_watchlist(body: dict):
        symbols = body.get("symbols", [])
        validated = state.coin_manager.set_watchlist(symbols)
        return {"ok": True, "watchlist": validated}

    @app.get("/api/watchlist/overview")
    async def get_watchlist_overview():
        try:
            overview = await asyncio.to_thread(state.coin_manager.get_watchlist_overview)
            return {"ok": True, "coins": overview}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Signal (active coin) ──────────────────────────────

    @app.get("/api/signal")
    async def get_signal():
        try:
            data = await asyncio.to_thread(state.coin_manager.refresh_active_signal)
            return {"ok": True, "data": data}
        except Exception as e:
            logger.exception("Signal refresh error")
            return {"ok": False, "error": str(e)}

    @app.get("/api/signal/cached")
    async def get_cached_signal():
        return {"ok": True, "data": state.coin_manager.active_signal_data}

    @app.post("/api/interval")
    async def set_interval(body: dict):
        interval = body.get("interval", "1h")
        state.coin_manager.set_interval(interval)
        return {"ok": True, "interval": interval}

    # ── LLM Providers ─────────────────────────────────────

    @app.get("/api/providers")
    async def get_providers():
        return state.llm_manager.get_status()

    @app.post("/api/settings")
    async def update_settings(body: dict):
        state.llm_manager.configure(
            body.get("provider", ""),
            body.get("api_key", ""),
            body.get("model", ""),
            body.get("enabled", True),
        )
        return {"ok": True, "status": state.llm_manager.get_status()}

    # ── WebSocket ─────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        logger.info("WebSocket connected")
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                t = msg.get("type", "")

                if t == "chat":
                    await _handle_chat(ws, state, msg)
                elif t == "discussion":
                    await _handle_discussion(ws, state, msg)
                elif t == "refresh":
                    await _handle_refresh(ws, state)
                elif t == "select_coin":
                    await _handle_select_coin(ws, state, msg)
                elif t == "refresh_watchlist":
                    await _handle_refresh_watchlist(ws, state)
                elif t == "set_watchlist":
                    await _handle_set_watchlist(ws, state, msg)
                else:
                    await ws.send_json({"type": "error", "message": f"Unknown: {t}"})
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception:
            logger.exception("WebSocket error")

    return app


# ── WebSocket Handlers ────────────────────────────────────

async def _handle_refresh(ws: WebSocket, state: BTCDumpWebApp) -> None:
    await ws.send_json({"type": "status", "message": "Refreshing signal..."})
    try:
        await asyncio.to_thread(state.coin_manager.refresh_active_signal)
        await ws.send_json({
            "type": "signal_data",
            "data": state.coin_manager.active_signal_data,
        })
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})


async def _handle_select_coin(ws: WebSocket, state: BTCDumpWebApp, msg: dict) -> None:
    symbol = msg.get("symbol", "").upper()
    state.coin_manager.set_active_coin(symbol)
    await ws.send_json({"type": "status", "message": f"Switched to {symbol}..."})
    try:
        data = await asyncio.to_thread(state.coin_manager.compute_signal, symbol)
        await ws.send_json({"type": "coin_selected", "symbol": symbol, "signal_data": data})
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})


async def _handle_refresh_watchlist(ws: WebSocket, state: BTCDumpWebApp) -> None:
    """Compute signals for all watchlist coins, streaming updates."""
    watchlist = state.coin_manager.watchlist
    total = len(watchlist)
    await ws.send_json({
        "type": "watchlist_progress",
        "data": {"completed": 0, "total": total, "current_symbol": ""},
    })

    completed = 0
    for symbol in watchlist:
        try:
            data = await asyncio.to_thread(state.coin_manager.compute_signal, symbol)
            mini = await asyncio.to_thread(state.coin_manager.get_mini_chart_data, symbol)
            data["mini_chart"] = mini
        except Exception as exc:
            data = {"symbol": symbol, "error": str(exc), "status": "error"}
            mini = []

        completed += 1
        await ws.send_json({
            "type": "watchlist_update",
            "data": {**data, "mini_chart": mini},
        })
        await ws.send_json({
            "type": "watchlist_progress",
            "data": {"completed": completed, "total": total, "current_symbol": symbol},
        })

    await ws.send_json({"type": "status", "message": "Watchlist refresh complete"})


async def _handle_set_watchlist(ws: WebSocket, state: BTCDumpWebApp, msg: dict) -> None:
    symbols = msg.get("symbols", [])
    validated = state.coin_manager.set_watchlist(symbols)
    await ws.send_json({"type": "status", "message": f"Watchlist: {len(validated)} coins"})


async def _handle_chat(ws: WebSocket, state: BTCDumpWebApp, msg: dict) -> None:
    provider_name = msg.get("provider", "")
    user_message = msg.get("message", "")
    if not user_message:
        return

    provider = state.llm_manager.get_provider(provider_name)
    if not provider:
        await ws.send_json({
            "type": "chat_chunk", "provider": provider_name,
            "content": f"[{provider_name} not configured. Add API key in Settings.]",
            "done": True,
        })
        return

    from btcdump.web.discussion import _build_market_context

    # Build context-aware system prompt
    ctx = msg.get("context", {})
    mode = ctx.get("mode", "single")
    cm = state.coin_manager

    if mode == "compare":
        context_text = cm.get_compare_context()
        system_msg = (
            f"You are an AI crypto analyst. The user is comparing multiple coins. "
            f"Be concise and data-driven.\n\n{context_text}"
        )
    else:
        symbol = ctx.get("symbol", cm.active_symbol)
        display = symbol.replace("USDT", "/USDT")
        interval = ctx.get("interval", cm.active_interval)
        signal_data = cm.signal_cache.get(symbol, cm.active_signal_data)
        system_msg = (
            f"You are an AI crypto analyst for {display} ({interval}). "
            f"Be concise and specific.\n\n"
            f"{_build_market_context(signal_data)}"
        )

    history = state.chat_histories.get(provider_name, [])
    history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history[-10:])

    full_response = []
    try:
        async for chunk in provider.complete_stream(messages):
            full_response.append(chunk)
            await ws.send_json({
                "type": "chat_chunk", "provider": provider_name,
                "content": chunk, "done": False,
            })
    except Exception as e:
        full_response.append(f"[Error: {e}]")
        await ws.send_json({
            "type": "chat_chunk", "provider": provider_name,
            "content": f"[Error: {e}]", "done": False,
        })

    await ws.send_json({
        "type": "chat_chunk", "provider": provider_name,
        "content": "", "done": True,
    })
    history.append({"role": "assistant", "content": "".join(full_response)})
    state.chat_histories[provider_name] = history


async def _handle_discussion(ws: WebSocket, state: BTCDumpWebApp, msg: dict) -> None:
    question = msg.get("message", "")
    num_rounds = msg.get("rounds", 3)
    if not question:
        return

    await ws.send_json({
        "type": "discussion_start", "question": question, "rounds": num_rounds,
    })

    async def on_chunk(provider, round_num, content, done):
        await ws.send_json({
            "type": "discussion_chunk", "provider": provider,
            "round": round_num, "content": content, "done": done,
        })

    await state.discussion.run_discussion(
        question=question,
        signal_data=state.coin_manager.active_signal_data,
        num_rounds=num_rounds,
        on_chunk=on_chunk,
    )
    await ws.send_json({"type": "discussion_complete"})


def run_server(host: str = "0.0.0.0", port: int = 8000, config: Optional[AppConfig] = None):
    import uvicorn
    app = create_app(config)
    print(f"\n  BTCDump Web UI starting at http://localhost:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
