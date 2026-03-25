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
from btcdump.web.alerts import AlertManager
from btcdump.web.coin_manager import CoinManager
from btcdump.web.discussion import DiscussionEngine
from btcdump.web.live_feed import BinanceLiveFeed
from btcdump.web.llm import LLMManager, PROVIDER_MODELS
from btcdump.web.notifications import NotificationManager
from btcdump.web.paper_trading import PaperTrader

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

        self.chat_histories: Dict[str, list] = {p: [] for p in PROVIDER_MODELS}
        self.notifications = NotificationManager()
        self.alerts = AlertManager()
        self.paper_trader = PaperTrader()
        self.live_feed = BinanceLiveFeed()
        self.connected_ws: set = set()

        ensure_dirs(self.config.data.cache_dir, self.config.model.models_dir)
        setup_logging(self.config.log_level, self.config.log_file)


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    state = BTCDumpWebApp(config)
    app = FastAPI(title="BTCDump", version="5.0.0")
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

    # ── Feature Importance ────────────────────────────────

    FEATURE_DESC = {
        "close":"Closing price","volume":"Trading volume","RSI":"Relative Strength Index (14)","MACD":"MACD line (12-26 EMA)","volume_ratio":"Volume / 20-period avg","ma5":"5-period SMA","ma20":"20-period SMA","ma50":"50-period SMA","BB_upper":"Upper Bollinger Band","BB_lower":"Lower Bollinger Band","ATR":"Average True Range (14)","stoch_k":"Stochastic %K (14)","stoch_d":"Stochastic %D (3)","ADX":"Avg Directional Index (14)","OBV_norm":"On-Balance Volume z-score","ROC":"Rate of Change (10)","williams_r":"Williams %R (14)","CCI":"Commodity Channel Index (20)","MFI":"Money Flow Index (14)","returns_1":"1-candle return","returns_5":"5-candle return","price_momentum":"10-bar price momentum","volatility_10":"10-period return volatility","high_low_range":"Candle range % of price","parkinson_vol":"Parkinson volatility","atr_ratio":"ATR expansion ratio","body_ratio":"Candle body ratio (0=doji,1=full)","buying_pressure":"Close position in range","upper_shadow":"Upper wick ratio","lower_shadow":"Lower wick ratio","bb_pct_b":"Bollinger %B position","bb_width":"BB width (squeeze)","vol_delta":"Volume delta (buy/sell)","ad_norm":"Accumulation/Distribution","hh_streak":"Higher highs streak","ll_streak":"Lower lows streak","dist_from_high":"Distance from 20-bar high","dist_from_low":"Distance from 20-bar low","rsi_momentum":"RSI acceleration","macd_hist_slope":"MACD histogram slope","close_ma_ratio":"Price vs MA20 ratio","price_zscore":"Price z-score","vwap_dist":"VWAP distance in ATR",
        # v5.0 Pro Features
        "efficiency_ratio":"Kaufman ER - trend efficiency (0=chop,1=clean trend)","choppiness":"Choppiness Index (>61.8=chop,<38.2=trend)","adx_slope":"ADX 5-bar change (trend strengthening/weakening)","tsi":"True Strength Index (double-smoothed momentum)","rsi_divergence":"Price-RSI divergence (positive=bullish div)","returns_10":"10-candle return","returns_20":"20-candle return","momentum_quality":"Directional Sharpe (return/volatility)","garch_proxy":"GARCH proxy (short/long vol ratio, >1=expanding)","vol_of_vol":"Volatility of volatility (regime transition signal)","yang_zhang_vol":"Yang-Zhang vol (state-of-art OHLCV estimator)","volume_trend":"Volume slope (rising=accumulation)","amihud_illiq":"Amihud illiquidity (higher=less liquid)","hour_sin":"Hour sine (cyclical time feature)","hour_cos":"Hour cosine (cyclical time feature)","dow_sin":"Day-of-week sine (weekend/weekday cycle)","dow_cos":"Day-of-week cosine","skewness_20":"Return skewness (neg=left tail risk)","kurtosis_20":"Return kurtosis (high=extreme moves likely)","keltner_position":"Keltner Channel position (ATR-based bands)","squeeze_ratio":"BB/Keltner squeeze (<1=squeeze,breakout imminent)","engulfing_score":"Engulfing pattern strength (pos=bullish)","consecutive_dir":"Consecutive candle direction streak","ofi_14":"Order Flow Imbalance 14-bar (buy vs sell pressure proxy)","pv_divergence":"Price-Volume divergence (high=thin move, low=confirmed)","ichimoku_tk":"Ichimoku Tenkan-Kijun cross (% of price, pos=bullish)","ichimoku_cloud_pos":"Price position vs Ichimoku cloud (>0=above)","ichimoku_cloud_width":"Ichimoku cloud thickness (% of price, thin=weak S/R)","ichimoku_chikou":"Chikou span: price vs 26 bars ago (% change)","ichimoku_kijun_dist":"Distance from Kijun-sen baseline (mean-reversion)","vp_poc_dist":"Distance from Volume POC (Point of Control, %)","vp_va_position":"Position within Value Area (0=low, 1=high, >1=above)",
    }

    @app.get("/api/feature-importance")
    async def get_feature_importance():
        ens = state.coin_manager.active_ensemble
        if not ens or not ens.feature_importances:
            return {"ok": False, "error": "No model trained yet. Run Refresh first."}
        pairs = list(zip(ens.feature_names, ens.feature_importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        features = []
        for rank, (name, imp) in enumerate(pairs, 1):
            features.append({"name": name, "importance": round(imp, 5), "rank": rank, "description": FEATURE_DESC.get(name, name)})
        return {"ok": True, "features": features}

    # ── Multi-Timeframe ───────────────────────────────────

    @app.get("/api/coin/{symbol}/multi-tf")
    async def get_multi_tf(symbol: str):
        try:
            data = await asyncio.to_thread(state.coin_manager.compute_multi_tf_signal, symbol)
            return {"ok": True, **data}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Correlation Matrix ──────────────────────────────────

    @app.get("/api/correlation")
    async def get_correlation():
        try:
            data = await asyncio.to_thread(state.coin_manager.compute_correlation_matrix)
            return {"ok": True, **data}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Support / Resistance ─────────────────────────────

    @app.get("/api/coin/{symbol}/sr-levels")
    async def get_sr_levels(symbol: str):
        try:
            def _compute():
                from btcdump.indicators import detect_support_resistance
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                return detect_support_resistance(data.df)
            levels = await asyncio.to_thread(_compute)
            return {"ok": True, "levels": levels, "symbol": symbol}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Fear & Greed Index ───────────────────────────────

    @app.get("/api/fear-greed")
    async def get_fear_greed():
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get("https://api.alternative.me/fng/?limit=1")
                d = r.json()
                entry = d.get("data", [{}])[0]
                return {
                    "ok": True,
                    "value": int(entry.get("value", 50)),
                    "classification": entry.get("value_classification", "Neutral"),
                    "timestamp": entry.get("timestamp", ""),
                }
        except Exception as e:
            return {"ok": True, "value": 50, "classification": "Unavailable", "error": str(e)}

    # ── Anomaly Detection ────────────────────────────────

    @app.get("/api/coin/{symbol}/anomalies")
    async def get_anomalies(symbol: str):
        try:
            def _detect():
                from btcdump.indicators import detect_anomalies
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                return detect_anomalies(data.df)
            result = await asyncio.to_thread(_detect)
            return {"ok": True, **result, "symbol": symbol}
        except Exception as e:
            return {"ok": False, "error": str(e)}

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

    # ── Paper Trading ─────────────────────────────────────

    @app.post("/api/paper/open")
    async def paper_open(body: dict):
        try:
            sym = body.get("symbol", state.coin_manager.active_symbol).upper()
            side = body.get("side", "long")
            size = body.get("size_pct", 10)
            sl = body.get("stop_loss", 0)
            tp = body.get("take_profit", 0)
            tickers = {t["symbol"]: t["lastPrice"] for t in state.coin_manager.fetcher.fetch_tickers()}
            price = tickers.get(sym, 0)
            if not price:
                return {"ok": False, "error": f"No price for {sym}"}
            result = state.paper_trader.open_position(sym, side, price, size, sl, tp)
            return {"ok": True, "position": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/paper/close")
    async def paper_close(body: dict):
        try:
            sym = body.get("symbol", "").upper()
            tickers = {t["symbol"]: t["lastPrice"] for t in state.coin_manager.fetcher.fetch_tickers()}
            price = tickers.get(sym, 0)
            result = state.paper_trader.close_position(sym, price)
            return {"ok": True, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/paper/portfolio")
    async def paper_portfolio():
        tickers = {t["symbol"]: t["lastPrice"] for t in state.coin_manager.fetcher.fetch_tickers()}
        return {"ok": True, **state.paper_trader.get_portfolio(tickers)}

    @app.get("/api/paper/history")
    async def paper_history():
        return {"ok": True, "trades": state.paper_trader.get_history()}

    @app.post("/api/paper/reset")
    async def paper_reset():
        state.paper_trader.reset()
        return {"ok": True}

    # ── Alerts ────────────────────────────────────────────

    @app.post("/api/alerts")
    async def add_alert(body: dict):
        a = state.alerts.add(body.get("symbol", ""), body.get("condition", ""), body.get("value", 0))
        return {"ok": True, "alert": {"id": a.id, "symbol": a.symbol, "condition": a.condition, "value": a.value}}

    @app.get("/api/alerts")
    async def get_alerts():
        return {"ok": True, "alerts": state.alerts.get_all()}

    @app.delete("/api/alerts/{alert_id}")
    async def delete_alert(alert_id: str):
        return {"ok": state.alerts.remove(alert_id)}

    # ── Notifications ─────────────────────────────────────

    @app.post("/api/notifications/configure")
    async def configure_notifications(body: dict):
        state.notifications.configure(
            telegram_token=body.get("telegram_token", ""),
            telegram_chat_id=body.get("telegram_chat_id", ""),
            discord_webhook=body.get("discord_webhook", ""),
            enabled=body.get("enabled", True),
        )
        return {"ok": True, "status": state.notifications.get_status()}

    @app.get("/api/notifications/status")
    async def get_notification_status():
        return {"ok": True, **state.notifications.get_status()}

    @app.post("/api/notifications/test")
    async def test_notification():
        test_data = {"direction": "TEST", "confidence": 99, "current_price": 70000,
                     "predicted_price": 71000, "change_pct": 1.43, "rsi": 55, "risk_reward": 2.1}
        state.notifications._previous_signals["TEST"] = "HOLD"
        msg = await state.notifications.check_and_notify("TESTUSDT", {**test_data, "direction": "BUY"})
        return {"ok": bool(msg), "message": msg or "Notifications not configured"}

    # ── WebSocket ─────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        state.connected_ws.add(ws)
        logger.info("WebSocket connected (%d clients)", len(state.connected_ws))

        # Start live feed if not running
        async def on_tick(tick):
            for client in list(state.connected_ws):
                try:
                    await client.send_json({"type": "live_price", "data": tick})
                except Exception:
                    state.connected_ws.discard(client)
            # Check alerts on each tick
            triggered = state.alerts.check(tick["symbol"], tick["price"])
            for a in triggered:
                for client in list(state.connected_ws):
                    try:
                        await client.send_json({"type": "alert_triggered", "alert": {
                            "id": a.id, "symbol": a.symbol,
                            "condition": a.condition, "value": a.value,
                        }})
                    except Exception:
                        pass

        symbols = list(set([state.coin_manager.active_symbol] + state.coin_manager.watchlist))
        for s in symbols:
            state.live_feed.subscribe(s, on_tick)
        if not state.live_feed._task:
            await state.live_feed.start(symbols)

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
                elif t == "run_backtest":
                    await _handle_backtest(ws, state, msg)
                else:
                    await ws.send_json({"type": "error", "message": f"Unknown: {t}"})
        except WebSocketDisconnect:
            state.connected_ws.discard(ws)
            state.live_feed.unsubscribe_all(on_tick)
            logger.info("WebSocket disconnected (%d clients)", len(state.connected_ws))
        except Exception:
            state.connected_ws.discard(ws)
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


async def _handle_backtest(ws: WebSocket, state: BTCDumpWebApp, msg: dict) -> None:
    """Run backtest with progressive updates."""
    from btcdump.backtest import BacktestEngine

    symbol = msg.get("symbol", state.coin_manager.active_symbol)
    retrain_every = msg.get("retrain_every", 50)

    await ws.send_json({"type": "status", "message": f"Running backtest for {symbol}..."})

    engine = BacktestEngine(state.config)

    last_pct = [0]

    def on_progress(step, total):
        nonlocal last_pct
        pct = int(step / total * 100) if total > 0 else 0
        if pct >= last_pct[0] + 5:  # send every 5%
            last_pct[0] = pct
            import asyncio as _aio
            try:
                _aio.get_event_loop().create_task(
                    ws.send_json({"type": "backtest_progress", "data": {"step": step, "total": total, "pct": pct}})
                )
            except Exception:
                pass

    try:
        data = state.coin_manager.fetcher.fetch_with_cache(symbol, state.coin_manager.active_interval)
        result = await asyncio.to_thread(
            engine.run, data.df, symbol, state.coin_manager.active_interval, retrain_every,
        )

        await ws.send_json({"type": "backtest_complete", "data": {
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "total_return_pct": result.total_return_pct,
            "total_signals": result.total_signals,
            "avg_win_pct": result.avg_win_pct,
            "avg_loss_pct": result.avg_loss_pct,
            "signal_accuracy": result.signal_accuracy,
            "equity_curve": result.equity_curve[["equity", "drawdown"]].values.tolist(),
            "optimal_thresholds": result.optimal_thresholds,
        }})
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Backtest failed: {e}"})


def run_server(host: str = "0.0.0.0", port: int = 8000, config: Optional[AppConfig] = None):
    import uvicorn
    app = create_app(config)
    print(f"\n  BTCDump Web UI starting at http://localhost:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
