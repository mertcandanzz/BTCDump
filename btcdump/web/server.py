"""FastAPI server: REST + WebSocket for BTCDump multi-coin Web UI."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from btcdump import indicators
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
from btcdump.web.signal_history import SignalHistory

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
        self.signal_history = SignalHistory(self.config.data.cache_dir.parent)
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
            # Record signal in history
            if data.get("status") == "ready" and data.get("direction"):
                state.signal_history.record(data)
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
    async def get_ohlcv(symbol: str, limit: int = 100, chart_type: str = "candlestick"):
        """Full OHLCV data for candlestick chart. chart_type: candlestick or heikin_ashi."""
        try:
            def _fetch():
                interval = state.coin_manager.active_interval
                data = state.coin_manager.fetcher.fetch_with_cache(symbol, interval)
                df = data.df.tail(limit + 1)  # +1 for HA calculation

                if chart_type == "heikin_ashi":
                    # Heikin-Ashi transformation
                    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
                    ha_open = df["open"].copy()
                    for i in range(1, len(df)):
                        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
                    ha_high = df[["high"]].join(ha_open.rename("ha_o")).join(ha_close.rename("ha_c")).max(axis=1)
                    ha_low = df[["low"]].join(ha_open.rename("ha_o")).join(ha_close.rename("ha_c")).min(axis=1)

                    df = df.copy()
                    df["open"] = ha_open
                    df["high"] = ha_high
                    df["low"] = ha_low
                    df["close"] = ha_close

                df = df.tail(limit)
                return [{
                    "t": int(r["time"].timestamp() * 1000),
                    "o": round(float(r["open"]), 6),
                    "h": round(float(r["high"]), 6),
                    "l": round(float(r["low"]), 6),
                    "c": round(float(r["close"]), 6),
                    "v": round(float(r["volume"]), 2),
                } for _, r in df.iterrows()]
            candles = await asyncio.to_thread(_fetch)
            return {"ok": True, "candles": candles, "symbol": symbol, "chart_type": chart_type}
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
        "efficiency_ratio":"Kaufman ER - trend efficiency (0=chop,1=clean trend)","choppiness":"Choppiness Index (>61.8=chop,<38.2=trend)","adx_slope":"ADX 5-bar change (trend strengthening/weakening)","tsi":"True Strength Index (double-smoothed momentum)","rsi_divergence":"Price-RSI divergence (positive=bullish div)","returns_10":"10-candle return","returns_20":"20-candle return","momentum_quality":"Directional Sharpe (return/volatility)","garch_proxy":"GARCH proxy (short/long vol ratio, >1=expanding)","vol_of_vol":"Volatility of volatility (regime transition signal)","yang_zhang_vol":"Yang-Zhang vol (state-of-art OHLCV estimator)","volume_trend":"Volume slope (rising=accumulation)","amihud_illiq":"Amihud illiquidity (higher=less liquid)","hour_sin":"Hour sine (cyclical time feature)","hour_cos":"Hour cosine (cyclical time feature)","dow_sin":"Day-of-week sine (weekend/weekday cycle)","dow_cos":"Day-of-week cosine","skewness_20":"Return skewness (neg=left tail risk)","kurtosis_20":"Return kurtosis (high=extreme moves likely)","keltner_position":"Keltner Channel position (ATR-based bands)","squeeze_ratio":"BB/Keltner squeeze (<1=squeeze,breakout imminent)","engulfing_score":"Engulfing pattern strength (pos=bullish)","consecutive_dir":"Consecutive candle direction streak","ofi_14":"Order Flow Imbalance 14-bar (buy vs sell pressure proxy)","pv_divergence":"Price-Volume divergence (high=thin move, low=confirmed)","ichimoku_tk":"Ichimoku Tenkan-Kijun cross (% of price, pos=bullish)","ichimoku_cloud_pos":"Price position vs Ichimoku cloud (>0=above)","ichimoku_cloud_width":"Ichimoku cloud thickness (% of price, thin=weak S/R)","ichimoku_chikou":"Chikou span: price vs 26 bars ago (% change)","ichimoku_kijun_dist":"Distance from Kijun-sen baseline (mean-reversion)","vp_poc_dist":"Distance from Volume POC (Point of Control, %)","vp_va_position":"Position within Value Area (0=low, 1=high, >1=above)","pivot_dist":"Distance from pivot point (% of price)","pivot_r1_dist":"Distance to R1 resistance (%)","pivot_s1_dist":"Distance to S1 support (%)","pivot_position":"Position within S2-R2 range (0=S2, 1=R2)","pattern_doji":"Doji strength (1=perfect doji, indecision)","pattern_hammer":"Hammer/HangingMan strength (bullish reversal)","pattern_shooting_star":"Shooting Star strength (bearish reversal)","pattern_three_soldiers":"Three White Soldiers (strong bull continuation)","pattern_three_crows":"Three Black Crows (strong bear continuation)","pattern_morning_star":"Morning Star (bullish reversal, 3-candle)","pattern_evening_star":"Evening Star (bearish reversal, 3-candle)","trade_intensity":"Volume per unit price move z-score (high=absorption)","pin_bar_score":"Pin bar strength (pos=bullish, neg=bearish reversal)","gap_pct":"Gap % from prev close to open (crypto overnight gaps)","intrabar_vol_ratio":"Total range / body size (high=choppy intrabar)","close_position_avg":"5-bar avg close position in range (persistent pressure)","whale_score":"Whale activity score (high vol + small body = absorption)","smart_money_div":"Smart money divergence (1=bullish, -1=bearish accumulation/distribution)","price_entropy":"Shannon entropy of returns (high=uncertain/random, low=predictable)","kama_dist":"KAMA distance (Kaufman Adaptive MA, adapts to regime)","dema_dist":"DEMA distance (Double EMA, faster trend response)","tema_dist":"TEMA distance (Triple EMA, minimal lag)","kama_slope":"KAMA 3-bar slope (adaptive trend direction)","seasonal_hour_bias":"Hour-of-day seasonal return bias (z-score)","seasonal_dow_bias":"Day-of-week seasonal return bias (z-score)","hurst_exponent":"Hurst exponent (>0.5=trending, <0.5=mean-reverting, 0.5=random)","autocorr_1":"Return auto-correlation lag-1 (pos=momentum, neg=reversal)","autocorr_5":"Return auto-correlation lag-5 (weekly pattern on 1h data)","di_ratio":"DI+/DI- ratio (>1=bullish dominance, <1=bearish)","di_spread":"DI+/DI- normalized spread (-1 to +1)","variance_ratio":"Lo-MacKinlay variance ratio (>1=trending, <1=mean-reverting)",
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

    # ── Market Scanner ────────────────────────────────────

    @app.get("/api/scanner")
    async def market_scanner(condition: str = "rsi_oversold", limit: int = 20):
        """Scan top coins for trading conditions.

        Conditions: rsi_oversold, rsi_overbought, macd_cross_bull, macd_cross_bear,
        volume_spike, bollinger_squeeze, strong_trend, ichimoku_bull, ichimoku_bear
        """
        try:
            results = await asyncio.to_thread(
                _run_scanner, state, condition, limit,
            )
            return {"ok": True, "condition": condition, "results": results}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Trade Setup Generator ──────────────────────────────

    @app.get("/api/coin/{symbol}/trade-setup")
    async def get_trade_setup(symbol: str, capital: float = 10000, risk_pct: float = 1.0):
        """Generate a complete actionable trade setup."""
        try:
            def _generate():
                from btcdump import indicators as ind
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                enriched = ind.compute_all(data.df.copy(), state.config.indicators)
                row = enriched.iloc[-1]
                price = float(data.df["close"].iloc[-1])
                atr = float(row.get("ATR", 0))

                cached = state.coin_manager.signal_cache.get(symbol, {})
                direction = cached.get("direction", "HOLD")
                confidence = cached.get("confidence", 0)
                rsi = float(row.get("RSI", 50))

                is_long = "BUY" in direction
                is_short = "SELL" in direction

                if not atr or direction == "HOLD":
                    return {"action": "NO TRADE", "reason": "Signal is HOLD or insufficient data"}

                # Dynamic SL/TP based on ATR and regime
                efficiency = float(row.get("efficiency_ratio", 0.5))
                # Tighter stops in choppy markets, wider in trending
                sl_mult = 1.0 + efficiency  # 1.0-2.0x ATR
                tp_mult = sl_mult * 2.0     # Always 2:1 minimum R/R

                if is_long:
                    entry = price
                    sl = price - atr * sl_mult
                    tp1 = price + atr * tp_mult * 0.5   # Partial TP at 1R
                    tp2 = price + atr * tp_mult          # Full TP at 2R
                    tp3 = price + atr * tp_mult * 1.5    # Runner at 3R
                else:
                    entry = price
                    sl = price + atr * sl_mult
                    tp1 = price - atr * tp_mult * 0.5
                    tp2 = price - atr * tp_mult
                    tp3 = price - atr * tp_mult * 1.5

                # Position sizing (risk-based)
                risk_amount = capital * (risk_pct / 100)
                sl_distance = abs(entry - sl)
                position_size = risk_amount / sl_distance if sl_distance > 0 else 0
                position_value = position_size * entry
                leverage_needed = position_value / capital if capital > 0 else 0

                # S/R levels for context
                sr = ind.detect_support_resistance(data.df)
                supports = [s for s in sr if s["type"] == "support" and s["price"] < price]
                resistances = [s for s in sr if s["type"] == "resistance" and s["price"] > price]

                # Fibonacci
                fibs = ind.compute_fibonacci_levels(data.df)
                nearest_fibs = sorted(fibs, key=lambda f: abs(f["distance_pct"]))[:3]

                # Checklist
                checks = []
                if confidence >= 60:
                    checks.append({"item": "ML Confidence > 60%", "pass": True, "value": f"{confidence:.0f}%"})
                else:
                    checks.append({"item": "ML Confidence > 60%", "pass": False, "value": f"{confidence:.0f}%"})

                agreement = cached.get("model_agreement", 0) * 100
                checks.append({"item": "Model Agreement > 70%", "pass": agreement > 70, "value": f"{agreement:.0f}%"})

                vol_ratio = float(row.get("volume_ratio", 1))
                checks.append({"item": "Volume Confirmation > 1x", "pass": vol_ratio > 1, "value": f"{vol_ratio:.1f}x"})

                adx = float(row.get("ADX", 0))
                checks.append({"item": "Trend Strength (ADX > 20)", "pass": adx > 20, "value": f"{adx:.0f}"})

                rsi_ok = (is_long and rsi < 70) or (is_short and rsi > 30)
                checks.append({"item": "RSI Not Extreme", "pass": rsi_ok, "value": f"{rsi:.0f}"})

                passed = sum(1 for c in checks if c["pass"])
                grade = "A" if passed >= 4 else "B" if passed >= 3 else "C" if passed >= 2 else "D"

                return {
                    "action": f"{'LONG' if is_long else 'SHORT'} {symbol.replace('USDT','/USDT')}",
                    "direction": direction,
                    "grade": grade,
                    "confidence": round(confidence, 1),
                    "entry": round(entry, 6),
                    "stop_loss": round(sl, 6),
                    "tp1": round(tp1, 6),
                    "tp2": round(tp2, 6),
                    "tp3": round(tp3, 6),
                    "sl_distance_pct": round(sl_distance / entry * 100, 2),
                    "rr_ratio": round(abs(tp2 - entry) / sl_distance, 1) if sl_distance > 0 else 0,
                    "position_size": round(position_size, 6),
                    "position_value": round(position_value, 2),
                    "risk_amount": round(risk_amount, 2),
                    "leverage": round(leverage_needed, 1),
                    "atr": round(atr, 6),
                    "regime": "trending" if efficiency > 0.5 else "choppy",
                    "nearest_support": round(supports[0]["price"], 6) if supports else None,
                    "nearest_resistance": round(resistances[0]["price"], 6) if resistances else None,
                    "fibonacci": nearest_fibs,
                    "checklist": checks,
                    "notes": (
                        f"{'Trending' if efficiency > 0.5 else 'Choppy'} market (ER={efficiency:.2f}). "
                        f"ATR=${atr:.2f}. RSI={rsi:.0f}. "
                        f"{'Strong volume' if vol_ratio > 1.5 else 'Normal volume' if vol_ratio > 0.8 else 'Low volume'}."
                    ),
                }

            result = await asyncio.to_thread(_generate)
            return {"ok": True, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── CSV Export ───────────────────────────────────────

    @app.get("/api/export/paper-trades")
    async def export_paper_trades():
        """Export paper trading history as CSV."""
        from starlette.responses import Response
        trades = state.paper_trader.get_history()
        if not trades:
            return {"ok": False, "error": "No trades to export"}
        header = "symbol,side,entry_price,exit_price,pnl,pnl_pct,entry_time,exit_time\n"
        rows = [f"{t['symbol']},{t['side']},{t['entry']},{t['exit']},{t['pnl']},{t['pnl_pct']},{t['entry_time']},{t['exit_time']}" for t in trades]
        csv = header + "\n".join(rows)
        return Response(content=csv, media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=paper_trades.csv"})

    @app.get("/api/export/signal-history")
    async def export_signal_history():
        """Export signal history as CSV."""
        from starlette.responses import Response
        records = state.signal_history.get_history(limit=500)
        if not records:
            return {"ok": False, "error": "No history to export"}
        header = "timestamp,symbol,direction,confidence,price,predicted_price,change_pct,rsi,outcome,price_1h,price_4h,price_24h\n"
        rows = []
        for r in records:
            rows.append(f"{r.get('timestamp','')},{r.get('symbol','')},{r.get('direction','')},{r.get('confidence',0)},{r.get('price_at_signal',0)},{r.get('predicted_price',0)},{r.get('predicted_change_pct',0)},{r.get('rsi',0)},{r.get('outcome','')},{r.get('price_after_1h','')},{r.get('price_after_4h','')},{r.get('price_after_24h','')}")
        csv = header + "\n".join(rows)
        return Response(content=csv, media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=signal_history.csv"})

    # ── Signal Leaderboard ────────────────────────────────

    @app.get("/api/leaderboard")
    async def get_leaderboard():
        """Rank all cached coins by composite signal quality score."""
        cache = state.coin_manager.signal_cache
        if not cache:
            return {"ok": True, "coins": []}

        scored = []
        for symbol, data in cache.items():
            if data.get("status") != "ready" or "error" in data:
                continue

            direction = data.get("direction", "HOLD")
            confidence = data.get("confidence", 0)
            rr = data.get("risk_reward", 0)
            agreement = data.get("model_agreement", 0) * 100
            confluence = data.get("indicator_confluence", 0)
            volume_ratio = data.get("volume_ratio", 1)

            # Composite score: weighted sum of quality signals
            dir_mult = 1.0 if "STRONG" in direction else 0.7 if direction != "HOLD" else 0.3
            score = (
                confidence * 0.30 +        # ML confidence
                agreement * 0.20 +          # Model agreement
                confluence * 10 * 0.15 +    # Indicator alignment (0-5 scaled to 0-50)
                min(rr, 5) * 10 * 0.15 +    # Risk/Reward (capped at 5)
                min(volume_ratio, 3) * 15 * 0.10 +  # Volume confirmation
                dir_mult * 30 * 0.10        # Direction strength bonus
            )

            scored.append({
                "symbol": symbol,
                "baseAsset": symbol.replace("USDT", ""),
                "direction": direction,
                "confidence": round(confidence, 1),
                "risk_reward": round(rr, 2),
                "model_agreement": round(agreement, 1),
                "volume_ratio": round(volume_ratio, 2),
                "score": round(score, 1),
                "change_pct": round(data.get("change_pct", 0), 2),
                "rsi": data.get("rsi", 0),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"ok": True, "coins": scored[:20]}

    # ── Fibonacci Retracement ──────────────────────────────

    @app.get("/api/coin/{symbol}/fibonacci")
    async def get_fibonacci(symbol: str, lookback: int = 100):
        try:
            def _calc():
                from btcdump.indicators import compute_fibonacci_levels
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                return compute_fibonacci_levels(data.df, lookback)
            levels = await asyncio.to_thread(_calc)
            return {"ok": True, "levels": levels, "symbol": symbol}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Market Breadth ───────────────────────────────────

    @app.get("/api/market-breadth")
    async def get_market_breadth():
        """Aggregate health of watchlist coins."""
        cache = state.coin_manager.signal_cache
        watchlist = state.coin_manager.watchlist
        total = len(watchlist)
        if not total:
            return {"ok": True, "total": 0}

        bullish = sum(1 for s in watchlist if "BUY" in cache.get(s, {}).get("direction", ""))
        bearish = sum(1 for s in watchlist if "SELL" in cache.get(s, {}).get("direction", ""))
        neutral = total - bullish - bearish

        rsi_values = [cache.get(s, {}).get("rsi", 0) for s in watchlist if cache.get(s, {}).get("rsi")]
        avg_rsi = round(sum(rsi_values) / len(rsi_values), 1) if rsi_values else 0

        confidences = [cache.get(s, {}).get("confidence", 0) for s in watchlist if cache.get(s, {}).get("confidence")]
        avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0

        return {
            "ok": True,
            "total": total,
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "bullish_pct": round(bullish / total * 100),
            "bearish_pct": round(bearish / total * 100),
            "avg_rsi": avg_rsi,
            "avg_confidence": avg_conf,
            "sentiment": "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral",
        }

    # ── Signal History ─────────────────────────────────────

    @app.get("/api/signal-history")
    async def get_signal_history(symbol: str = "", limit: int = 50):
        records = state.signal_history.get_history(symbol, limit)
        return {"ok": True, "records": records}

    @app.get("/api/signal-history/stats")
    async def get_signal_stats(symbol: str = ""):
        stats = state.signal_history.get_stats(symbol)
        return {"ok": True, **stats}

    @app.post("/api/signal-history/update-outcomes")
    async def update_signal_outcomes():
        try:
            tickers = {t["symbol"]: t["lastPrice"] for t in state.coin_manager.fetcher.fetch_tickers()}
            updated = state.signal_history.update_outcomes(tickers)
            return {"ok": True, "updated": updated}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Dynamic SL/TP Calculator ─────────────────────────

    @app.get("/api/coin/{symbol}/sl-tp")
    async def get_sl_tp(symbol: str, atr_mult_sl: float = 1.5, atr_mult_tp: float = 2.5):
        """Calculate dynamic stop-loss and take-profit levels."""
        try:
            def _calc():
                from btcdump import indicators as ind
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                enriched = ind.compute_all(data.df.copy(), state.config.indicators)
                row = enriched.iloc[-1]
                price = float(data.df["close"].iloc[-1])
                atr = float(row.get("ATR", 0))
                rsi = float(row.get("RSI", 50))

                # Determine bias from cached signal
                cached = state.coin_manager.signal_cache.get(symbol, {})
                direction = cached.get("direction", "HOLD")
                is_long = "BUY" in direction

                if is_long:
                    sl = price - atr * atr_mult_sl
                    tp = price + atr * atr_mult_tp
                else:
                    sl = price + atr * atr_mult_sl
                    tp = price - atr * atr_mult_tp

                # Also get S/R levels for context
                sr = ind.detect_support_resistance(data.df)
                nearest_support = next(
                    (s["price"] for s in sr if s["type"] == "support" and s["price"] < price), None,
                )
                nearest_resistance = next(
                    (s["price"] for s in sr if s["type"] == "resistance" and s["price"] > price), None,
                )

                rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0

                return {
                    "price": round(price, 6),
                    "atr": round(atr, 6),
                    "direction": direction,
                    "stop_loss": round(sl, 6),
                    "take_profit": round(tp, 6),
                    "sl_distance_pct": round(abs(price - sl) / price * 100, 2),
                    "tp_distance_pct": round(abs(tp - price) / price * 100, 2),
                    "risk_reward": round(rr, 2),
                    "nearest_support": round(nearest_support, 6) if nearest_support else None,
                    "nearest_resistance": round(nearest_resistance, 6) if nearest_resistance else None,
                    "atr_mult_sl": atr_mult_sl,
                    "atr_mult_tp": atr_mult_tp,
                }

            result = await asyncio.to_thread(_calc)
            return {"ok": True, **result}
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

    # ── Relative Strength vs BTC ───────────────────────────

    @app.get("/api/coin/{symbol}/relative-strength")
    async def get_relative_strength(symbol: str):
        try:
            def _calc():
                from btcdump.indicators import compute_relative_strength
                interval = state.coin_manager.active_interval
                pair_data = state.coin_manager.fetcher.fetch_with_cache(symbol, interval)
                btc_data = state.coin_manager.fetcher.fetch_with_cache("BTCUSDT", interval)
                return compute_relative_strength(pair_data.df, btc_data.df)
            result = await asyncio.to_thread(_calc)
            return {"ok": True, "symbol": symbol, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Trend Lines ──────────────────────────────────────

    @app.get("/api/coin/{symbol}/trend-lines")
    async def get_trend_lines(symbol: str):
        try:
            def _calc():
                from btcdump.indicators import detect_trend_lines
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                return detect_trend_lines(data.df)
            lines = await asyncio.to_thread(_calc)
            return {"ok": True, "lines": lines, "symbol": symbol}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Performance Dashboard ────────────────────────────

    @app.get("/api/performance")
    async def get_performance():
        """Overall platform performance stats."""
        stats = state.signal_history.get_stats()
        paper = state.paper_trader.get_portfolio(
            {t["symbol"]: t["lastPrice"]
             for t in state.coin_manager.fetcher.fetch_tickers()}
        ) if state.paper_trader.history else {}

        return {
            "ok": True,
            "signals": stats,
            "paper_trading": {
                "total_trades": paper.get("total_trades", 0),
                "win_rate": paper.get("win_rate", 0),
                "total_pnl": paper.get("total_pnl", 0),
                "total_pnl_pct": paper.get("total_pnl_pct", 0),
            } if paper else {},
            "platform": {
                "features": len(state.config.features.feature_columns),
                "dimensions": len(state.config.features.feature_columns) * state.config.features.window_size,
                "watchlist_size": len(state.coin_manager.watchlist),
                "cached_signals": len(state.coin_manager.signal_cache),
                "active_alerts": len(state.alerts.alerts),
            },
        }

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

    # ── DCA Simulator ─────────────────────────────────────

    @app.get("/api/coin/{symbol}/dca-simulate")
    async def dca_simulate(symbol: str, amount: float = 100, frequency: str = "weekly"):
        """Simulate Dollar Cost Averaging on historical data."""
        try:
            def _simulate():
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                df = data.df.copy()

                # Determine step size based on frequency and interval
                interval = state.coin_manager.active_interval
                if frequency == "daily":
                    step = {"1h": 24, "4h": 6, "1d": 1, "15m": 96, "30m": 48, "5m": 288}.get(interval, 24)
                elif frequency == "weekly":
                    step = {"1h": 168, "4h": 42, "1d": 7, "15m": 672, "30m": 336, "5m": 2016}.get(interval, 168)
                else:  # monthly
                    step = {"1h": 720, "4h": 180, "1d": 30, "15m": 2880}.get(interval, 720)

                step = min(step, len(df) // 3)  # at least 3 purchases
                if step < 1:
                    step = 1

                total_invested = 0.0
                total_coins = 0.0
                purchases = []
                equity_curve = []

                for i in range(0, len(df), step):
                    price = float(df["close"].iloc[i])
                    if price <= 0:
                        continue
                    coins_bought = amount / price
                    total_invested += amount
                    total_coins += coins_bought
                    current_value = total_coins * price
                    purchases.append({
                        "price": round(price, 6),
                        "coins": round(coins_bought, 8),
                        "invested": round(total_invested, 2),
                        "value": round(current_value, 2),
                    })
                    equity_curve.append([round(current_value, 2), round(total_invested, 2)])

                current_price = float(df["close"].iloc[-1])
                current_value = total_coins * current_price
                avg_price = total_invested / total_coins if total_coins > 0 else 0
                pnl = current_value - total_invested
                pnl_pct = (pnl / total_invested * 100) if total_invested > 0 else 0

                # Compare with lump sum (all money at start)
                start_price = float(df["close"].iloc[0])
                lump_coins = total_invested / start_price if start_price > 0 else 0
                lump_value = lump_coins * current_price
                lump_pnl_pct = ((lump_value - total_invested) / total_invested * 100) if total_invested > 0 else 0

                return {
                    "total_invested": round(total_invested, 2),
                    "current_value": round(current_value, 2),
                    "total_coins": round(total_coins, 8),
                    "avg_buy_price": round(avg_price, 6),
                    "current_price": round(current_price, 6),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "num_purchases": len(purchases),
                    "frequency": frequency,
                    "amount_per_buy": amount,
                    "lump_sum_pnl_pct": round(lump_pnl_pct, 2),
                    "dca_vs_lump": round(pnl_pct - lump_pnl_pct, 2),
                    "equity_curve": equity_curve[-50:],  # last 50 points
                }

            result = await asyncio.to_thread(_simulate)
            return {"ok": True, "symbol": symbol, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Portfolio Optimizer ───────────────────────────────

    @app.get("/api/portfolio-optimize")
    async def portfolio_optimize():
        """Markowitz mean-variance portfolio optimization for watchlist."""
        try:
            def _optimize():
                import pandas as _pdo
                watchlist = state.coin_manager.watchlist
                if len(watchlist) < 2:
                    return {"error": "Need 2+ coins"}

                interval = state.coin_manager.active_interval
                returns_map = {}
                for sym in watchlist:
                    try:
                        data = state.coin_manager.fetcher.fetch_with_cache(sym, interval)
                        ret = data.df["close"].pct_change().dropna().tail(200)
                        returns_map[sym] = ret.values
                    except Exception:
                        continue

                if len(returns_map) < 2:
                    return {"error": "Need price data for 2+ coins"}

                min_len = min(len(v) for v in returns_map.values())
                df = _pdo.DataFrame({k: v[-min_len:] for k, v in returns_map.items()})

                mean_returns = df.mean() * 252  # annualized (rough)
                cov_matrix = df.cov() * 252
                n = len(df.columns)
                symbols = list(df.columns)

                # Monte Carlo: random portfolios
                best_sharpe = -999
                best_weights = None
                min_vol_val = 999
                min_vol_weights = None

                np_opt = _np
                for _ in range(5000):
                    w = np_opt.random.random(n)
                    w /= w.sum()

                    port_ret = float(np_opt.dot(w, mean_returns))
                    port_vol = float(np_opt.sqrt(np_opt.dot(w.T, np_opt.dot(cov_matrix.values, w))))
                    sharpe = port_ret / port_vol if port_vol > 0 else 0

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_weights = w
                    if port_vol < min_vol_val:
                        min_vol_val = port_vol
                        min_vol_weights = w

                # Equal weight baseline
                eq_w = np_opt.ones(n) / n
                eq_ret = float(np_opt.dot(eq_w, mean_returns))
                eq_vol = float(np_opt.sqrt(np_opt.dot(eq_w.T, np_opt.dot(cov_matrix.values, eq_w))))

                result = {
                    "max_sharpe": {
                        "weights": {symbols[i]: round(float(best_weights[i]), 4) for i in range(n)},
                        "expected_return": round(float(np_opt.dot(best_weights, mean_returns)) * 100, 2),
                        "volatility": round(float(np_opt.sqrt(np_opt.dot(best_weights.T, np_opt.dot(cov_matrix.values, best_weights)))) * 100, 2),
                        "sharpe_ratio": round(best_sharpe, 3),
                    },
                    "min_volatility": {
                        "weights": {symbols[i]: round(float(min_vol_weights[i]), 4) for i in range(n)},
                        "expected_return": round(float(np_opt.dot(min_vol_weights, mean_returns)) * 100, 2),
                        "volatility": round(min_vol_val * 100, 2),
                    },
                    "equal_weight": {
                        "expected_return": round(eq_ret * 100, 2),
                        "volatility": round(eq_vol * 100, 2),
                        "sharpe_ratio": round(eq_ret / eq_vol if eq_vol > 0 else 0, 3),
                    },
                    "symbols": symbols,
                }
                return result

            result = await asyncio.to_thread(_optimize)
            if "error" in result:
                return {"ok": False, **result}
            return {"ok": True, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Signal Calibration ───────────────────────────────

    @app.get("/api/signal-calibration")
    async def signal_calibration():
        """Analyze if confidence scores are well-calibrated."""
        records = state.signal_history.get_history(limit=500)
        resolved = [r for r in records if r.get("outcome") in ("correct", "wrong")]

        if len(resolved) < 10:
            return {"ok": False, "error": "Need 10+ resolved signals for calibration"}

        # Bucket by confidence ranges
        buckets = {}
        for r in resolved:
            conf = r.get("confidence", 50)
            bucket = int(conf // 10) * 10  # 0-10, 10-20, etc.
            bucket_key = f"{bucket}-{bucket+10}"
            if bucket_key not in buckets:
                buckets[bucket_key] = {"total": 0, "correct": 0}
            buckets[bucket_key]["total"] += 1
            if r["outcome"] == "correct":
                buckets[bucket_key]["correct"] += 1

        calibration = {}
        for k, v in sorted(buckets.items()):
            if v["total"] >= 3:
                actual_accuracy = v["correct"] / v["total"] * 100
                # Expected accuracy is the midpoint of the bucket
                mid = int(k.split("-")[0]) + 5
                calibration[k] = {
                    "expected": mid,
                    "actual": round(actual_accuracy, 1),
                    "count": v["total"],
                    "gap": round(actual_accuracy - mid, 1),  # positive = overconfident
                    "well_calibrated": abs(actual_accuracy - mid) < 15,
                }

        # Overall calibration score
        if calibration:
            avg_gap = sum(abs(v["gap"]) for v in calibration.values()) / len(calibration)
            cal_score = max(0, round(100 - avg_gap * 2, 1))
        else:
            cal_score = 0

        return {
            "ok": True,
            "calibration": calibration,
            "calibration_score": cal_score,
            "interpretation": (
                "Well-calibrated" if cal_score > 70 else
                "Moderately calibrated" if cal_score > 50 else
                "Poorly calibrated - confidence scores need adjustment"
            ),
            "total_analyzed": len(resolved),
        }

    # ── Funding Rate ─────────────────────────────────────

    @app.get("/api/coin/{symbol}/funding-rate")
    async def get_funding_rate(symbol: str):
        """Get current funding rate from Binance Futures."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=10"
                )
                data = r.json()
                if not data:
                    return {"ok": False, "error": "No funding data (spot only?)"}

                latest = data[-1] if data else {}
                rate = float(latest.get("fundingRate", 0))
                annual = rate * 3 * 365  # 8h intervals, 3 per day

                history = [{
                    "rate": round(float(d.get("fundingRate", 0)) * 100, 4),
                    "time": d.get("fundingTime", 0),
                } for d in data]

                return {
                    "ok": True,
                    "symbol": symbol,
                    "current_rate": round(rate * 100, 4),
                    "annual_rate": round(annual * 100, 2),
                    "sentiment": "bearish (shorts paying)" if rate > 0.01 else "bullish (longs paying)" if rate < -0.01 else "neutral",
                    "history": history,
                }
        except Exception as e:
            return {"ok": True, "symbol": symbol, "current_rate": 0, "annual_rate": 0,
                    "sentiment": "unavailable", "history": [], "note": str(e)}

    # ── Volatility Term Structure ──────────────────────────

    @app.get("/api/coin/{symbol}/vol-term-structure")
    async def vol_term_structure(symbol: str):
        """Compare realized volatility across multiple timeframes."""
        try:
            def _calc():
                results = {}
                for tf in ["5m", "15m", "1h", "4h", "1d"]:
                    try:
                        data = state.coin_manager.fetcher.fetch_with_cache(symbol, tf)
                        ret = data.df["close"].pct_change().dropna()
                        if len(ret) < 20:
                            continue
                        # Annualized vol (rough: multiply by sqrt of periods per year)
                        periods_map = {"5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}
                        ann_factor = np.sqrt(periods_map.get(tf, 8760))
                        vol = float(ret.tail(50).std() * ann_factor * 100)
                        vol_short = float(ret.tail(10).std() * ann_factor * 100)
                        results[tf] = {
                            "realized_vol": round(vol, 2),
                            "short_term_vol": round(vol_short, 2),
                            "vol_ratio": round(vol_short / vol, 3) if vol > 0 else 1,
                        }
                    except Exception:
                        continue

                # Term structure shape
                vols = [results[tf]["realized_vol"] for tf in ["15m", "1h", "4h", "1d"] if tf in results]
                if len(vols) >= 2:
                    if vols[-1] > vols[0] * 1.1:
                        shape = "contango (higher TF vol > lower = normal)"
                    elif vols[-1] < vols[0] * 0.9:
                        shape = "backwardation (lower TF vol > higher = stress/event)"
                    else:
                        shape = "flat (similar vol across TFs)"
                else:
                    shape = "insufficient data"

                return {"timeframes": results, "shape": shape}

            result = await asyncio.to_thread(_calc)
            return {"ok": True, "symbol": symbol, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Market Health Score ───────────────────────────────

    @app.get("/api/coin/{symbol}/market-health")
    async def market_health(symbol: str):
        """Composite market quality/health score from microstructure data."""
        try:
            def _calc():
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                enriched = indicators.compute_all(data.df.copy(), state.config.indicators)
                row = enriched.iloc[-1]

                def _safe(key, default=0):
                    v = row.get(key, default)
                    return default if (hasattr(v, '__float__') and v != v) else float(v)

                # Components (each 0-100)
                # 1. Liquidity (volume ratio + Amihud)
                vol_ratio = min(3, _safe("volume_ratio", 1))
                liquidity = min(100, vol_ratio * 33)

                # 2. Trend clarity (efficiency + ADX)
                efficiency = _safe("efficiency_ratio", 0.5)
                adx = _safe("ADX", 25)
                trend_clarity = min(100, efficiency * 60 + adx * 1.0)

                # 3. Volatility regime (not too high, not too low)
                garch = _safe("garch_proxy", 1)
                # Optimal vol ratio around 1.0; too high or low is bad
                vol_quality = max(0, 100 - abs(garch - 1) * 60)

                # 4. Order flow (buying vs selling balance)
                ofi = abs(_safe("ofi_14", 0))
                flow_strength = min(100, ofi * 200)

                # 5. Spread/microstructure (trade intensity, entropy)
                entropy = _safe("price_entropy", 1.5)
                # Lower entropy = more predictable = healthier for trading
                predictability = max(0, 100 - entropy * 30)

                # 6. Whale activity (bonus)
                whale = _safe("whale_score", 0)
                whale_bonus = min(15, whale * 5)

                # Composite
                health = (
                    liquidity * 0.25 +
                    trend_clarity * 0.25 +
                    vol_quality * 0.20 +
                    flow_strength * 0.15 +
                    predictability * 0.15 +
                    whale_bonus
                )
                health = min(100, max(0, health))

                if health >= 70:
                    grade, interpretation = "A", "Excellent trading conditions"
                elif health >= 55:
                    grade, interpretation = "B", "Good conditions, proceed with normal sizing"
                elif health >= 40:
                    grade, interpretation = "C", "Fair conditions, reduce position size"
                elif health >= 25:
                    grade, interpretation = "D", "Poor conditions, consider sitting out"
                else:
                    grade, interpretation = "F", "Avoid trading - low liquidity or high chaos"

                return {
                    "health_score": round(health, 1),
                    "grade": grade,
                    "interpretation": interpretation,
                    "components": {
                        "liquidity": round(liquidity, 1),
                        "trend_clarity": round(trend_clarity, 1),
                        "vol_quality": round(vol_quality, 1),
                        "flow_strength": round(flow_strength, 1),
                        "predictability": round(predictability, 1),
                        "whale_bonus": round(whale_bonus, 1),
                    },
                }

            result = await asyncio.to_thread(_calc)
            return {"ok": True, "symbol": symbol, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Open Interest ─────────────────────────────────────

    @app.get("/api/coin/{symbol}/open-interest")
    async def get_open_interest(symbol: str):
        """Get open interest from Binance Futures."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                # Current OI
                r = await client.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}")
                current = r.json()

                # OI history (last 30 periods)
                r2 = await client.get(
                    f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=1h&limit=30"
                )
                history = r2.json() if r2.status_code == 200 else []

                oi = float(current.get("openInterest", 0))
                oi_values = [float(h.get("sumOpenInterest", 0)) for h in history] if history else []

                # OI change
                oi_change = 0
                if len(oi_values) >= 2:
                    oi_change = round((oi_values[-1] - oi_values[0]) / oi_values[0] * 100, 2) if oi_values[0] > 0 else 0

                return {
                    "ok": True,
                    "symbol": symbol,
                    "open_interest": oi,
                    "oi_change_pct": oi_change,
                    "interpretation": (
                        "Rising OI + Rising Price = New longs (bullish)" if oi_change > 5 else
                        "Falling OI + Rising Price = Short covering" if oi_change < -5 else
                        "Stable OI"
                    ),
                    "history": [{"oi": float(h.get("sumOpenInterest", 0)), "time": h.get("timestamp", 0)} for h in history[-10:]],
                }
        except Exception as e:
            return {"ok": True, "symbol": symbol, "open_interest": 0, "oi_change_pct": 0,
                    "interpretation": "unavailable", "history": [], "note": str(e)}

    # ── Market Summary ───────────────────────────────────

    @app.get("/api/market-summary")
    async def market_summary():
        """Quick market overview: BTC dominance proxy, total market sentiment."""
        try:
            tickers = state.coin_manager.fetcher.fetch_tickers()
            total_vol = sum(float(t.get("quoteVolume", 0)) for t in tickers)
            btc_vol = next((float(t.get("quoteVolume", 0)) for t in tickers if t["symbol"] == "BTCUSDT"), 0)
            eth_vol = next((float(t.get("quoteVolume", 0)) for t in tickers if t["symbol"] == "ETHUSDT"), 0)

            gainers = sum(1 for t in tickers if float(t.get("priceChangePercent", 0)) > 0)
            losers = len(tickers) - gainers
            avg_change = sum(float(t.get("priceChangePercent", 0)) for t in tickers) / len(tickers) if tickers else 0

            # Top gainers and losers
            sorted_tickers = sorted(tickers, key=lambda t: float(t.get("priceChangePercent", 0)), reverse=True)
            top_gainers = [{
                "symbol": t["symbol"], "baseAsset": t.get("baseAsset", ""),
                "change": round(float(t.get("priceChangePercent", 0)), 2),
            } for t in sorted_tickers[:5]]
            top_losers = [{
                "symbol": t["symbol"], "baseAsset": t.get("baseAsset", ""),
                "change": round(float(t.get("priceChangePercent", 0)), 2),
            } for t in sorted_tickers[-5:]]

            return {
                "ok": True,
                "total_pairs": len(tickers),
                "gainers": gainers,
                "losers": losers,
                "avg_change_pct": round(avg_change, 2),
                "market_sentiment": "bullish" if gainers > losers * 1.5 else "bearish" if losers > gainers * 1.5 else "neutral",
                "btc_volume_dominance": round(btc_vol / total_vol * 100, 1) if total_vol else 0,
                "top_gainers": top_gainers,
                "top_losers": top_losers,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Regime-Adaptive Signal ─────────────────────────────

    @app.get("/api/coin/{symbol}/signal-adaptive")
    async def get_adaptive_signal(symbol: str):
        """Generate signal with regime-adaptive thresholds."""
        try:
            def _compute():
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                ensemble = state.coin_manager.ensembles.get(symbol)
                if not ensemble:
                    ensemble = state.pipeline.load(symbol, state.coin_manager.active_interval)
                if not ensemble:
                    return None

                pred, conf, indiv = state.pipeline.predict(ensemble, data.df)
                enriched = indicators.compute_all(data.df.copy(), state.config.indicators)
                current_price = float(data.df["close"].iloc[-1])
                row = enriched.iloc[-1]

                # Standard signal
                standard = state.signal_gen.generate(
                    current_price, pred, conf, indiv, row,
                )

                # Regime-adaptive signal
                adaptive = state.signal_gen.generate_regime_adaptive(
                    current_price, pred, conf, indiv, row,
                )

                return {
                    "standard": {
                        "direction": standard.direction,
                        "confidence": standard.confidence,
                    },
                    "adaptive": {
                        "direction": adaptive.direction,
                        "confidence": adaptive.confidence,
                    },
                    "regime_info": {
                        "efficiency_ratio": round(float(row.get("efficiency_ratio", 0.5)), 3),
                        "adx": round(float(row.get("ADX", 25)), 1),
                        "hurst": round(float(row.get("hurst_exponent", 0.5)), 3),
                    },
                    "signals_agree": standard.direction == adaptive.direction,
                }

            result = await asyncio.to_thread(_compute)
            if not result:
                return {"ok": False, "error": "No model. Refresh first."}
            return {"ok": True, "symbol": symbol, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Smart Multi-Condition Alerts ──────────────────────

    @app.post("/api/smart-alerts")
    async def add_smart_alert(body: dict):
        """Create multi-condition alert.

        Body: {symbol, conditions: [{field, operator, value}, ...], name}
        Example: {symbol: "BTCUSDT", conditions: [
            {field: "rsi", operator: "<", value: 30},
            {field: "volume_ratio", operator: ">", value: 2}
        ], name: "RSI oversold + volume spike"}
        """
        symbol = body.get("symbol", "").upper()
        conditions = body.get("conditions", [])
        name = body.get("name", "Custom Alert")

        if not conditions:
            return {"ok": False, "error": "No conditions provided"}

        alert_id = f"smart-{symbol}-{int(datetime.now().timestamp())}"
        smart_alert = {
            "id": alert_id,
            "symbol": symbol,
            "name": name,
            "conditions": conditions,
            "created": datetime.now().isoformat(),
            "triggered": False,
        }

        if not hasattr(state, "_smart_alerts"):
            state._smart_alerts = []
        state._smart_alerts.append(smart_alert)

        return {"ok": True, "alert": smart_alert}

    @app.get("/api/smart-alerts")
    async def get_smart_alerts():
        alerts = getattr(state, "_smart_alerts", [])
        return {"ok": True, "alerts": alerts}

    @app.post("/api/smart-alerts/check")
    async def check_smart_alerts():
        """Check all smart alerts against current data."""
        alerts = getattr(state, "_smart_alerts", [])
        if not alerts:
            return {"ok": True, "triggered": []}

        triggered = []
        for alert in alerts:
            if alert.get("triggered"):
                continue
            symbol = alert["symbol"]
            cached = state.coin_manager.signal_cache.get(symbol, {})
            if not cached:
                continue

            all_met = True
            for cond in alert["conditions"]:
                field = cond.get("field", "")
                op = cond.get("operator", "")
                val = cond.get("value", 0)

                actual = cached.get(field, 0)
                if actual == 0 and field == "rsi":
                    actual = cached.get("rsi", 50)

                try:
                    actual = float(actual)
                    val = float(val)
                except (ValueError, TypeError):
                    all_met = False
                    break

                if op == ">" and not (actual > val):
                    all_met = False
                    break
                elif op == "<" and not (actual < val):
                    all_met = False
                    break
                elif op == ">=" and not (actual >= val):
                    all_met = False
                    break
                elif op == "<=" and not (actual <= val):
                    all_met = False
                    break
                elif op == "==" and not (abs(actual - val) < 0.01):
                    all_met = False
                    break

            if all_met:
                alert["triggered"] = True
                triggered.append(alert)

        return {"ok": True, "triggered": triggered}

    # ── Auto-Feature Analysis ────────────────────────────

    @app.get("/api/feature-analysis")
    async def feature_analysis():
        """Analyze feature quality: importance distribution, correlation clusters."""
        ens = state.coin_manager.active_ensemble
        if not ens or not ens.feature_importances:
            return {"ok": False, "error": "No model trained yet."}

        names = ens.feature_names
        importances = ens.feature_importances

        pairs = list(zip(names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)

        # Top features that contribute 80% of importance
        total = sum(importances)
        cumulative = 0
        core_features = []
        for name, imp in pairs:
            cumulative += imp
            core_features.append(name)
            if cumulative >= total * 0.8:
                break

        # Bottom features (< 0.5% each)
        noise_features = [n for n, i in pairs if i < total * 0.005]

        # Concentration: how concentrated is importance in top features
        top_10_share = sum(i for _, i in pairs[:10]) / total * 100 if total > 0 else 0

        return {
            "ok": True,
            "total_features": len(names),
            "core_features": core_features,
            "core_count": len(core_features),
            "noise_features": noise_features,
            "noise_count": len(noise_features),
            "top_10_share_pct": round(top_10_share, 1),
            "concentration": "high" if top_10_share > 60 else "moderate" if top_10_share > 40 else "distributed",
            "recommendation": (
                f"Top {len(core_features)} features explain 80% of predictions. "
                f"{len(noise_features)} features contribute <0.5% each and could be pruned "
                f"to reduce overfitting risk."
            ),
        }

    # ── Correlation Breakdown Detection ─────────────────────

    @app.get("/api/correlation-breakdown")
    async def correlation_breakdown():
        """Detect pairs where correlation has broken down from historical norm."""
        try:
            def _detect():
                import pandas as _pdd
                interval = state.coin_manager.active_interval
                watchlist = state.coin_manager.watchlist
                if len(watchlist) < 2:
                    return []

                # Get returns for all watchlist coins
                returns_map = {}
                for sym in watchlist:
                    try:
                        data = state.coin_manager.fetcher.fetch_with_cache(sym, interval)
                        returns_map[sym] = data.df["close"].pct_change().dropna()
                    except Exception:
                        continue

                if len(returns_map) < 2:
                    return []

                breakdowns = []
                symbols = list(returns_map.keys())

                for i in range(len(symbols)):
                    for j in range(i + 1, len(symbols)):
                        s1, s2 = symbols[i], symbols[j]
                        r1, r2 = returns_map[s1], returns_map[s2]

                        # Align
                        min_len = min(len(r1), len(r2))
                        if min_len < 50:
                            continue
                        r1 = r1.tail(min_len).values
                        r2 = r2.tail(min_len).values

                        # Long-term correlation (full history)
                        long_corr = float(_pdd.Series(r1).corr(_pdd.Series(r2)))
                        # Short-term correlation (last 20 bars)
                        short_corr = float(_pdd.Series(r1[-20:]).corr(_pdd.Series(r2[-20:])))

                        # Breakdown = big difference between long and short term
                        diff = abs(long_corr - short_corr)
                        if diff > 0.3 and abs(long_corr) > 0.4:
                            breakdowns.append({
                                "pair": f"{s1.replace('USDT','')}-{s2.replace('USDT','')}",
                                "symbol1": s1,
                                "symbol2": s2,
                                "long_term_corr": round(long_corr, 3),
                                "short_term_corr": round(short_corr, 3),
                                "divergence": round(diff, 3),
                                "interpretation": (
                                    f"Normally {'positively' if long_corr > 0 else 'negatively'} correlated "
                                    f"({long_corr:.2f}), now {'decorrelating' if abs(short_corr) < abs(long_corr) * 0.5 else 'shifting'}. "
                                    f"Watch for mean-reversion or regime change."
                                ),
                            })

                breakdowns.sort(key=lambda x: x["divergence"], reverse=True)
                return breakdowns[:5]

            result = await asyncio.to_thread(_detect)
            return {"ok": True, "breakdowns": result, "count": len(result)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Seasonality Profile ────────────────────────────────

    @app.get("/api/coin/{symbol}/seasonality")
    async def get_seasonality(symbol: str):
        try:
            def _calc():
                from btcdump.indicators import compute_seasonality_profile
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                return compute_seasonality_profile(data.df)
            result = await asyncio.to_thread(_calc)
            return {"ok": True, "symbol": symbol, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Monte Carlo Simulation ───────────────────────────

    @app.get("/api/monte-carlo")
    async def monte_carlo(simulations: int = 1000, horizon: int = 30):
        """Monte Carlo simulation of portfolio outcomes."""
        try:
            import numpy as np_mc

            trades = state.paper_trader.get_history()
            if len(trades) < 3:
                # Use signal history if no trades
                records = state.signal_history.get_history(limit=100)
                returns = []
                for r in records:
                    if r.get("price_after_24h") and r.get("price_at_signal"):
                        ret = (r["price_after_24h"] - r["price_at_signal"]) / r["price_at_signal"]
                        returns.append(ret)
                if not returns:
                    returns = [0.001, -0.001, 0.002, -0.0015, 0.0005]  # default
            else:
                returns = [t["pnl_pct"] / 100 for t in trades]

            returns = np_mc.array(returns)
            mu = float(returns.mean())
            sigma = float(returns.std()) or 0.01

            balance = 10000
            simulations = min(simulations, 5000)

            # Run simulations
            final_values = []
            paths_sample = []  # keep 10 sample paths
            for i in range(simulations):
                path = [balance]
                val = balance
                for _ in range(horizon):
                    ret = np_mc.random.normal(mu, sigma)
                    val *= (1 + ret)
                    path.append(round(float(val), 2))
                final_values.append(float(val))
                if i < 10:
                    paths_sample.append(path)

            final_arr = np_mc.array(final_values)
            percentiles = {
                "p5": round(float(np_mc.percentile(final_arr, 5)), 2),
                "p25": round(float(np_mc.percentile(final_arr, 25)), 2),
                "p50": round(float(np_mc.percentile(final_arr, 50)), 2),
                "p75": round(float(np_mc.percentile(final_arr, 75)), 2),
                "p95": round(float(np_mc.percentile(final_arr, 95)), 2),
            }

            return {
                "ok": True,
                "simulations": simulations,
                "horizon_trades": horizon,
                "initial_balance": balance,
                "mean_final": round(float(final_arr.mean()), 2),
                "std_final": round(float(final_arr.std()), 2),
                "percentiles": percentiles,
                "prob_profit": round(float((final_arr > balance).mean()) * 100, 1),
                "prob_double": round(float((final_arr > balance * 2).mean()) * 100, 1),
                "prob_ruin": round(float((final_arr < balance * 0.5).mean()) * 100, 1),
                "sample_paths": paths_sample,
                "avg_return_per_trade": round(mu * 100, 3),
                "std_per_trade": round(sigma * 100, 3),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Dashboard Summary ────────────────────────────────

    @app.get("/api/dashboard")
    async def get_dashboard():
        """One-page summary of all platform intelligence."""
        try:
            cache = state.coin_manager.signal_cache
            active = state.coin_manager.active_signal_data
            stats = state.signal_history.get_stats()

            # Best opportunity
            best = None
            best_score = 0
            for sym, data in cache.items():
                if data.get("status") != "ready":
                    continue
                conf = data.get("confidence", 0)
                dir_val = data.get("direction", "HOLD")
                score = conf * (1.5 if "STRONG" in dir_val else 1.0 if dir_val != "HOLD" else 0.3)
                if score > best_score:
                    best_score = score
                    best = data

            return {
                "ok": True,
                "active_coin": {
                    "symbol": state.coin_manager.active_symbol,
                    "direction": active.get("direction", "--"),
                    "confidence": active.get("confidence", 0),
                    "price": active.get("current_price", 0),
                },
                "best_opportunity": {
                    "symbol": best.get("symbol", "--") if best else "--",
                    "direction": best.get("direction", "--") if best else "--",
                    "confidence": best.get("confidence", 0) if best else 0,
                } if best else None,
                "signal_accuracy": stats.get("accuracy", 0),
                "total_signals": stats.get("total_signals", 0),
                "watchlist_size": len(state.coin_manager.watchlist),
                "cached_signals": len(cache),
                "features": len(state.config.features.feature_columns),
                "interval": state.coin_manager.active_interval,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Prediction Confidence Intervals ─────────────────────

    @app.get("/api/coin/{symbol}/prediction-range")
    async def get_prediction_range(symbol: str):
        """Get prediction with confidence intervals."""
        try:
            def _predict():
                ensemble = state.coin_manager.ensembles.get(symbol)
                if not ensemble:
                    ensemble = state.pipeline.load(symbol, state.coin_manager.active_interval)
                if not ensemble:
                    return None
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                return state.pipeline.predict_with_intervals(ensemble, data.df)

            result = await asyncio.to_thread(_predict)
            if not result:
                return {"ok": False, "error": "No trained model. Refresh signal first."}
            return {"ok": True, "symbol": symbol, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Momentum Rotation ─────────────────────────────────

    @app.get("/api/momentum-rotation")
    async def momentum_rotation():
        """Suggest coin rotation based on momentum + regime analysis."""
        try:
            def _rotate():
                from btcdump import indicators as ind
                interval = state.coin_manager.active_interval
                results = []

                try:
                    btc_data = state.coin_manager.fetcher.fetch_with_cache("BTCUSDT", interval)
                except Exception:
                    btc_data = None

                for symbol in state.coin_manager.watchlist:
                    try:
                        data = state.coin_manager.fetcher.fetch_with_cache(symbol, interval)
                        enriched = ind.compute_all(data.df.copy(), state.config.indicators)
                        row = enriched.iloc[-1]

                        # Momentum score (multi-timeframe momentum)
                        ret_1 = float(row.get("returns_1", 0)) * 100
                        ret_5 = float(row.get("returns_5", 0)) * 100
                        ret_10 = float(row.get("returns_10", 0)) * 100
                        ret_20 = float(row.get("returns_20", 0)) * 100

                        # Weighted momentum: recent > distant
                        momentum = ret_1 * 0.1 + ret_5 * 0.2 + ret_10 * 0.3 + ret_20 * 0.4

                        # Regime bonus: trending coins get boosted
                        efficiency = float(row.get("efficiency_ratio", 0.5))
                        hurst = float(row.get("hurst_exponent", 0.5))
                        regime_mult = 1.0 + max(0, efficiency - 0.4) + max(0, hurst - 0.5)

                        # Volume confirmation
                        vol_ratio = float(row.get("volume_ratio", 1))
                        vol_mult = min(1.5, max(0.5, vol_ratio))

                        # RS vs BTC
                        rs_bonus = 0
                        if btc_data and symbol != "BTCUSDT":
                            rs = ind.compute_relative_strength(data.df, btc_data.df)
                            rs_bonus = rs.get("rs_ratio", 0) * 0.3

                        final_score = momentum * regime_mult * vol_mult + rs_bonus

                        results.append({
                            "symbol": symbol,
                            "baseAsset": symbol.replace("USDT", ""),
                            "momentum_score": round(final_score, 2),
                            "momentum_raw": round(momentum, 2),
                            "regime": "trending" if efficiency > 0.5 else "choppy",
                            "volume": round(vol_ratio, 2),
                            "ret_1d": round(ret_20, 2),
                            "ret_5bar": round(ret_5, 2),
                            "rs_vs_btc": round(rs_bonus / 0.3, 2) if rs_bonus else 0,
                        })
                    except Exception:
                        continue

                results.sort(key=lambda x: x["momentum_score"], reverse=True)

                # Top 3 = BUY rotation, Bottom 3 = potential SELL/avoid
                buy_candidates = results[:3] if results else []
                sell_candidates = results[-3:] if len(results) > 3 else []

                return {
                    "all": results,
                    "buy_rotation": buy_candidates,
                    "sell_rotation": sell_candidates,
                    "summary": f"Rotate into: {', '.join(r['baseAsset'] for r in buy_candidates)}. "
                               f"Avoid: {', '.join(r['baseAsset'] for r in sell_candidates)}." if results else "Need watchlist signals first.",
                }

            result = await asyncio.to_thread(_rotate)
            return {"ok": True, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Risk Dashboard + Kelly Sizing ──────────────────────

    @app.get("/api/risk-dashboard")
    async def risk_dashboard():
        """Comprehensive risk analysis across portfolio and signals."""
        try:
            # Signal history stats
            stats = state.signal_history.get_stats()
            win_rate = stats.get("accuracy", 0) / 100 if stats.get("resolved") else 0.5

            # Kelly Criterion: f* = (bp - q) / b
            # b = win/loss ratio, p = win probability, q = loss probability
            by_dir = stats.get("by_direction", {})
            avg_win_pct = 1.5  # default assumption
            avg_loss_pct = 1.0

            # Calculate from paper trading if available
            trades = state.paper_trader.get_history()
            if len(trades) >= 5:
                wins = [t for t in trades if t["pnl"] > 0]
                losses = [t for t in trades if t["pnl"] <= 0]
                if wins:
                    avg_win_pct = sum(t["pnl_pct"] for t in wins) / len(wins)
                if losses:
                    avg_loss_pct = abs(sum(t["pnl_pct"] for t in losses) / len(losses))
                win_rate = len(wins) / len(trades)

            b = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 1
            p = win_rate
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0
            kelly = max(0, min(0.25, kelly))  # Cap at 25%
            half_kelly = kelly / 2  # Conservative Kelly

            # Portfolio risk (if positions open)
            portfolio = {}
            try:
                tickers = {t["symbol"]: t["lastPrice"] for t in state.coin_manager.fetcher.fetch_tickers()}
                portfolio = state.paper_trader.get_portfolio(tickers)
            except Exception:
                pass

            # Watchlist correlation risk
            cache = state.coin_manager.signal_cache
            directions = [cache.get(s, {}).get("direction", "") for s in state.coin_manager.watchlist]
            same_direction = max(
                sum(1 for d in directions if "BUY" in d),
                sum(1 for d in directions if "SELL" in d),
            )
            concentration_risk = round(same_direction / max(1, len(directions)) * 100)

            return {
                "ok": True,
                "kelly": {
                    "full_kelly_pct": round(kelly * 100, 2),
                    "half_kelly_pct": round(half_kelly * 100, 2),
                    "win_rate": round(win_rate * 100, 1),
                    "avg_win_pct": round(avg_win_pct, 2),
                    "avg_loss_pct": round(avg_loss_pct, 2),
                    "win_loss_ratio": round(b, 2),
                    "recommended_size_pct": round(half_kelly * 100, 2),
                    "recommended_size_usd": round(half_kelly * portfolio.get("total_value", 10000), 2),
                },
                "portfolio": {
                    "total_value": portfolio.get("total_value", 10000),
                    "open_positions": len(portfolio.get("positions", [])),
                    "total_pnl_pct": portfolio.get("total_pnl_pct", 0),
                },
                "risk_metrics": {
                    "concentration_risk_pct": concentration_risk,
                    "signals_same_direction": same_direction,
                    "total_watchlist": len(state.coin_manager.watchlist),
                    "total_signals_tracked": stats.get("total_signals", 0),
                    "signal_accuracy_pct": stats.get("accuracy", 0),
                },
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Webhook Signal Forwarding ─────────────────────────

    @app.post("/api/webhook/configure")
    async def configure_webhook(body: dict):
        """Configure webhook URL for signal forwarding."""
        url = body.get("url", "")
        state._webhook_url = url
        state._webhook_enabled = bool(url)
        return {"ok": True, "enabled": state._webhook_enabled, "url": url}

    @app.get("/api/webhook/status")
    async def webhook_status():
        return {
            "ok": True,
            "enabled": getattr(state, "_webhook_enabled", False),
            "url": getattr(state, "_webhook_url", ""),
        }

    @app.post("/api/webhook/test")
    async def test_webhook():
        url = getattr(state, "_webhook_url", "")
        if not url:
            return {"ok": False, "error": "No webhook URL configured"}
        try:
            import httpx
            payload = {
                "type": "btcdump_signal",
                "symbol": state.coin_manager.active_symbol,
                "direction": state.coin_manager.active_signal_data.get("direction", "HOLD"),
                "confidence": state.coin_manager.active_signal_data.get("confidence", 0),
                "price": state.coin_manager.active_signal_data.get("current_price", 0),
                "timestamp": datetime.now().isoformat() if True else "",
            }
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.post(url, json=payload)
                return {"ok": True, "status_code": r.status_code}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── AI Trade Narrative ────────────────────────────────

    @app.get("/api/coin/{symbol}/narrative")
    async def get_trade_narrative(symbol: str):
        """Generate comprehensive market analysis narrative from all indicators."""
        try:
            cached = state.coin_manager.signal_cache.get(symbol, {})
            if not cached or cached.get("status") != "ready":
                return {"ok": False, "error": "No signal data. Refresh first."}

            # Build comprehensive context
            direction = cached.get("direction", "HOLD")
            confidence = cached.get("confidence", 0)
            price = cached.get("current_price", 0)
            pred_price = cached.get("predicted_price", 0)
            rsi = cached.get("rsi", 0)
            adx = cached.get("adx", 0)
            macd_bull = cached.get("macd_bullish", False)
            vol_ratio = cached.get("volume_ratio", 1)
            rr = cached.get("risk_reward", 0)
            stoch = cached.get("stoch_k", 50)
            atr = cached.get("atr", 0)

            # Get regime
            regime_r = await get_market_regime(symbol)
            regime = regime_r.get("regime", "unknown") if regime_r.get("ok") else "unknown"
            strategy = regime_r.get("strategy", "") if regime_r.get("ok") else ""

            # Get RS
            rs_data = {}
            if symbol != "BTCUSDT":
                try:
                    rs_r = await get_relative_strength(symbol)
                    if rs_r.get("ok"):
                        rs_data = rs_r
                except Exception:
                    pass

            display = symbol.replace("USDT", "/USDT")
            interval = state.coin_manager.active_interval

            sections = []

            # Signal Summary
            conf_desc = "high" if confidence > 65 else "moderate" if confidence > 45 else "low"
            sections.append(f"**{display} ({interval}) - {direction}** (Confidence: {confidence:.0f}% - {conf_desc})")
            sections.append(f"Current: ${price:,.2f} | AI Prediction: ${pred_price:,.2f} ({cached.get('change_pct', 0):+.2f}%)")

            # Market Regime
            regime_emoji = {"trending_up": "📈", "trending_down": "📉", "breakout": "💥", "range": "↔️"}.get(regime, "❓")
            sections.append(f"\n**Market Regime:** {regime_emoji} {regime.replace('_',' ').title()} → {strategy}")

            # Technical Analysis
            ta = []
            if rsi < 30: ta.append(f"RSI={rsi:.0f} (OVERSOLD - potential reversal)")
            elif rsi > 70: ta.append(f"RSI={rsi:.0f} (OVERBOUGHT - potential reversal)")
            else: ta.append(f"RSI={rsi:.0f} ({'bullish zone' if rsi > 50 else 'bearish zone'})")

            ta.append(f"MACD: {'Bullish' if macd_bull else 'Bearish'} | ADX: {adx:.0f} ({'strong trend' if adx > 40 else 'developing' if adx > 20 else 'no trend'})")
            ta.append(f"Stochastic: {stoch:.0f} | Volume: {vol_ratio:.1f}x {'(HIGH)' if vol_ratio > 1.5 else '(normal)' if vol_ratio > 0.8 else '(LOW - caution)'}")
            sections.append("\n**Technical:**\n" + "\n".join(f"• {t}" for t in ta))

            # Relative Strength
            if rs_data:
                rs_val = rs_data.get("rs_ratio", 0)
                rs_class = rs_data.get("classification", "neutral")
                sections.append(f"\n**vs BTC:** {rs_val:+.2f}% ({rs_class.replace('_',' ')})")

            # Risk Assessment
            risk = []
            risk.append(f"ATR: ${atr:,.2f} | Risk/Reward: {rr:.1f}")
            agreement = cached.get("model_agreement", 0) * 100
            risk.append(f"Model Agreement: {agreement:.0f}% | Confluence: {cached.get('indicator_confluence', 0)}/5")
            sections.append("\n**Risk:**\n" + "\n".join(f"• {r}" for r in risk))

            # Actionable Conclusion
            if direction == "HOLD":
                sections.append("\n**Action:** WAIT. No clear edge. Monitor for regime change.")
            elif "STRONG" in direction:
                sections.append(f"\n**Action:** {'Enter LONG' if 'BUY' in direction else 'Enter SHORT'} with conviction. All systems aligned.")
            else:
                sections.append(f"\n**Action:** {'Consider LONG' if 'BUY' in direction else 'Consider SHORT'} with {'tight' if regime == 'range' else 'normal'} stops. {conf_desc.capitalize()} confidence.")

            narrative = "\n".join(sections)
            return {"ok": True, "narrative": narrative, "symbol": symbol}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Strategy Comparison ────────────────────────────────

    @app.get("/api/coin/{symbol}/strategy-compare")
    async def strategy_compare(symbol: str):
        """Compare different trading strategies on recent data."""
        try:
            def _compare():
                from btcdump import indicators as ind
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                enriched = ind.compute_all(data.df.copy(), state.config.indicators)
                df = enriched.tail(200)  # last 200 candles
                results = {}

                for strategy_name, signal_fn in _STRATEGIES.items():
                    signals = signal_fn(df)
                    pnl, trades, wins = _simulate_signals(df, signals)
                    total_ret = float(pnl[-1]) if len(pnl) else 0
                    win_rate = wins / trades if trades > 0 else 0
                    results[strategy_name] = {
                        "total_return_pct": round(total_ret, 2),
                        "trades": trades,
                        "win_rate": round(win_rate * 100, 1),
                        "avg_per_trade": round(total_ret / trades, 2) if trades > 0 else 0,
                    }

                return results

            result = await asyncio.to_thread(_compare)
            return {"ok": True, "symbol": symbol, "strategies": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Market Regime Classifier ───────────────────────────

    @app.get("/api/coin/{symbol}/regime")
    async def get_market_regime(symbol: str):
        """Classify current market regime into 4 states."""
        try:
            def _classify():
                from btcdump import indicators as ind
                data = state.coin_manager.fetcher.fetch_with_cache(
                    symbol, state.coin_manager.active_interval,
                )
                enriched = ind.compute_all(data.df.copy(), state.config.indicators)
                row = enriched.iloc[-1]

                adx = float(row.get("ADX", 0))
                efficiency = float(row.get("efficiency_ratio", 0.5))
                hurst = float(row.get("hurst_exponent", 0.5))
                choppiness = float(row.get("choppiness", 50))
                garch = float(row.get("garch_proxy", 1))
                rsi = float(row.get("RSI", 50))
                bb_width = float(row.get("bb_width", 0.02))
                squeeze = float(row.get("squeeze_ratio", 1))
                macd = float(row.get("MACD", 0))
                macd_sig = float(row.get("MACD_signal", 0))
                variance_ratio = float(row.get("variance_ratio", 1))

                # Scoring system for 4 regimes
                trending_score = 0
                trending_score += min(40, adx * 1.0)           # ADX > 25 = trending
                trending_score += efficiency * 30               # ER near 1 = trending
                trending_score += (hurst - 0.5) * 40           # H > 0.5 = trending
                trending_score += max(0, (50 - choppiness)) * 0.5  # Low choppiness

                # Direction
                bullish_score = 0
                if macd > macd_sig:
                    bullish_score += 25
                if rsi > 50:
                    bullish_score += 25
                ret_20 = float(row.get("returns_20", 0))
                bullish_score += min(25, max(-25, ret_20 * 500))

                # Volatility expansion
                vol_expansion_score = 0
                vol_expansion_score += max(0, (garch - 1)) * 30  # GARCH > 1 = expanding
                vol_expansion_score += max(0, (1 - squeeze)) * 40  # Squeeze ratio < 1
                vol_expansion_score += max(0, bb_width * 500)

                # Range-bound indicators
                range_score = 0
                range_score += max(0, (0.5 - efficiency)) * 60  # Low ER
                range_score += max(0, choppiness - 50) * 0.8    # High choppiness
                range_score += max(0, (1 - variance_ratio)) * 30  # VR < 1

                # Classify
                if trending_score > 50 and bullish_score > 40:
                    regime = "trending_up"
                    desc = "Strong uptrend with momentum. Favor long positions, trail stops."
                    strategy = "Trend Following (Long)"
                    color = "#26a69a"
                elif trending_score > 50 and bullish_score < 20:
                    regime = "trending_down"
                    desc = "Strong downtrend. Favor shorts or stay flat. Avoid catching knives."
                    strategy = "Trend Following (Short)"
                    color = "#ef5350"
                elif vol_expansion_score > 40 and squeeze < 0.8:
                    regime = "breakout"
                    desc = "Volatility expanding from squeeze. Watch for breakout direction confirmation."
                    strategy = "Breakout / Momentum"
                    color = "#ff9800"
                else:
                    regime = "range"
                    desc = "Choppy/sideways market. Mean-reversion strategies work best. Tight stops."
                    strategy = "Mean Reversion / Scalping"
                    color = "#7b68ee"

                return {
                    "regime": regime,
                    "description": desc,
                    "strategy": strategy,
                    "color": color,
                    "scores": {
                        "trending": round(trending_score, 1),
                        "bullish": round(bullish_score, 1),
                        "volatility": round(vol_expansion_score, 1),
                        "range": round(range_score, 1),
                    },
                    "indicators": {
                        "adx": round(adx, 1),
                        "efficiency_ratio": round(efficiency, 3),
                        "hurst": round(hurst, 3),
                        "choppiness": round(choppiness, 1),
                        "garch_proxy": round(garch, 2),
                        "squeeze_ratio": round(squeeze, 3),
                        "variance_ratio": round(variance_ratio, 3),
                    },
                }

            result = await asyncio.to_thread(_classify)
            return {"ok": True, "symbol": symbol, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Smart Ranking (best trade opportunities) ──────────

    @app.get("/api/smart-rank")
    async def smart_rank():
        """Rank watchlist coins by trade opportunity quality.

        Combines: signal direction, confidence, volume, regime,
        relative strength, and anomaly detection into one score.
        """
        try:
            def _rank():
                from btcdump import indicators as ind
                results = []
                interval = state.coin_manager.active_interval

                # Pre-fetch BTC data for RS calculation
                try:
                    btc_data = state.coin_manager.fetcher.fetch_with_cache("BTCUSDT", interval)
                except Exception:
                    btc_data = None

                for symbol in state.coin_manager.watchlist:
                    try:
                        cached = state.coin_manager.signal_cache.get(symbol, {})
                        if not cached or cached.get("status") != "ready":
                            continue

                        data = state.coin_manager.fetcher.fetch_with_cache(symbol, interval)
                        enriched = ind.compute_all(data.df.copy(), state.config.indicators)
                        row = enriched.iloc[-1]

                        # Component scores (0-100 each)
                        direction = cached.get("direction", "HOLD")
                        dir_score = 100 if "STRONG" in direction else 70 if direction != "HOLD" else 20

                        conf_score = min(100, cached.get("confidence", 0))

                        vol_ratio = float(row.get("volume_ratio", 1))
                        vol_score = min(100, vol_ratio * 40)

                        efficiency = float(row.get("efficiency_ratio", 0.5))
                        regime_score = efficiency * 100  # trending = higher score

                        # RS vs BTC
                        rs_score = 50
                        if btc_data and symbol != "BTCUSDT":
                            rs = ind.compute_relative_strength(data.df, btc_data.df)
                            rs_score = min(100, max(0, 50 + rs.get("rs_ratio", 0) * 10))

                        # Anomaly bonus (unusual activity = opportunity)
                        anomaly = ind.detect_anomalies(data.df)
                        anomaly_bonus = 15 if anomaly.get("volume_anomaly") else 0

                        # Whale bonus
                        whale = float(row.get("whale_score", 0))
                        whale_bonus = min(15, whale * 5) if whale > 0 else 0

                        total = (
                            dir_score * 0.25 +
                            conf_score * 0.25 +
                            vol_score * 0.15 +
                            regime_score * 0.15 +
                            rs_score * 0.10 +
                            anomaly_bonus +
                            whale_bonus
                        ) * 0.1  # scale to ~0-10

                        results.append({
                            "symbol": symbol,
                            "baseAsset": symbol.replace("USDT", ""),
                            "direction": direction,
                            "score": round(total, 1),
                            "confidence": round(cached.get("confidence", 0), 1),
                            "volume_ratio": round(vol_ratio, 2),
                            "regime": "trending" if efficiency > 0.5 else "choppy",
                            "rs_vs_btc": round(rs_score - 50, 1),
                            "whale_activity": bool(whale > 1),
                            "anomaly": bool(anomaly.get("volume_anomaly") or anomaly.get("price_anomaly")),
                            "rsi": round(float(row.get("RSI", 0)), 1),
                            "change_pct": round(cached.get("change_pct", 0), 2),
                        })
                    except Exception:
                        continue

                results.sort(key=lambda x: x["score"], reverse=True)
                return results

            ranked = await asyncio.to_thread(_rank)
            return {"ok": True, "coins": ranked}
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

    @app.post("/api/paper/journal")
    async def add_journal_note(body: dict):
        trade_id = body.get("trade_id", "")
        note = body.get("note", "")
        if not trade_id or not note:
            return {"ok": False, "error": "trade_id and note required"}
        entry = state.paper_trader.add_note(trade_id, note)
        return {"ok": True, "entry": entry}

    @app.get("/api/paper/journal")
    async def get_journal(trade_id: str = ""):
        return {"ok": True, **state.paper_trader.get_journal(trade_id)}

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


# ── Strategy Definitions for Comparison ────────────────────

import numpy as _np
import pandas as _pd


def _strat_rsi_reversion(df):
    """Buy when RSI < 30, sell when RSI > 70."""
    signals = _np.zeros(len(df))
    rsi = df.get("RSI", _pd.Series(50, index=df.index))
    for i in range(1, len(df)):
        if rsi.iloc[i] < 30:
            signals[i] = 1  # buy
        elif rsi.iloc[i] > 70:
            signals[i] = -1  # sell
    return signals


def _strat_macd_crossover(df):
    """Buy on MACD bullish cross, sell on bearish cross."""
    signals = _np.zeros(len(df))
    macd = df.get("MACD", _pd.Series(0, index=df.index))
    sig = df.get("MACD_signal", _pd.Series(0, index=df.index))
    for i in range(1, len(df)):
        if macd.iloc[i] > sig.iloc[i] and macd.iloc[i-1] <= sig.iloc[i-1]:
            signals[i] = 1
        elif macd.iloc[i] < sig.iloc[i] and macd.iloc[i-1] >= sig.iloc[i-1]:
            signals[i] = -1
    return signals


def _strat_bollinger_bounce(df):
    """Buy near lower BB, sell near upper BB."""
    signals = _np.zeros(len(df))
    c = df["close"]
    bb_lower = df.get("BB_lower", c)
    bb_upper = df.get("BB_upper", c)
    for i in range(1, len(df)):
        if c.iloc[i] <= bb_lower.iloc[i] * 1.005:
            signals[i] = 1
        elif c.iloc[i] >= bb_upper.iloc[i] * 0.995:
            signals[i] = -1
    return signals


def _strat_ema_trend(df):
    """Buy when EMA9 > EMA21, sell when EMA9 < EMA21."""
    signals = _np.zeros(len(df))
    c = df["close"]
    ema9 = c.ewm(span=9).mean()
    ema21 = c.ewm(span=21).mean()
    for i in range(1, len(df)):
        if ema9.iloc[i] > ema21.iloc[i] and ema9.iloc[i-1] <= ema21.iloc[i-1]:
            signals[i] = 1
        elif ema9.iloc[i] < ema21.iloc[i] and ema9.iloc[i-1] >= ema21.iloc[i-1]:
            signals[i] = -1
    return signals


def _strat_ichimoku(df):
    """Buy above cloud + TK cross, sell below cloud + TK cross."""
    signals = _np.zeros(len(df))
    cloud_pos = df.get("ichimoku_cloud_pos", _pd.Series(0, index=df.index))
    tk = df.get("ichimoku_tk", _pd.Series(0, index=df.index))
    for i in range(1, len(df)):
        if cloud_pos.iloc[i] > 0 and tk.iloc[i] > 0 and tk.iloc[i-1] <= 0:
            signals[i] = 1
        elif cloud_pos.iloc[i] < 0 and tk.iloc[i] < 0 and tk.iloc[i-1] >= 0:
            signals[i] = -1
    return signals


_STRATEGIES = {
    "RSI Mean Reversion": _strat_rsi_reversion,
    "MACD Crossover": _strat_macd_crossover,
    "Bollinger Bounce": _strat_bollinger_bounce,
    "EMA Trend Follow": _strat_ema_trend,
    "Ichimoku Cloud": _strat_ichimoku,
}


def _simulate_signals(df, signals):
    """Simple signal simulator: returns cumulative PnL, trade count, wins."""
    closes = df["close"].values
    pnl = []
    cum = 0.0
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0
    trades = 0
    wins = 0

    for i in range(len(signals)):
        if signals[i] == 1 and position <= 0:
            if position == -1:  # close short
                ret = (entry_price - closes[i]) / entry_price * 100
                cum += ret
                trades += 1
                if ret > 0: wins += 1
            position = 1
            entry_price = closes[i]
        elif signals[i] == -1 and position >= 0:
            if position == 1:  # close long
                ret = (closes[i] - entry_price) / entry_price * 100
                cum += ret
                trades += 1
                if ret > 0: wins += 1
            position = -1
            entry_price = closes[i]
        pnl.append(cum)

    # Close open position
    if position == 1:
        ret = (closes[-1] - entry_price) / entry_price * 100
        cum += ret
        trades += 1
        if ret > 0: wins += 1
    elif position == -1:
        ret = (entry_price - closes[-1]) / entry_price * 100
        cum += ret
        trades += 1
        if ret > 0: wins += 1

    pnl.append(cum)
    return pnl, trades, wins


def _run_scanner(state: BTCDumpWebApp, condition: str, limit: int) -> list:
    """Run market scanner across top coins for given condition."""
    from btcdump import indicators

    # Fetch top coins by volume
    try:
        tickers = state.coin_manager.fetcher.fetch_tickers()
    except Exception:
        return []

    # Sort by volume and take top candidates
    tickers.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
    candidates = [t["symbol"] for t in tickers[:60]]

    CONDITION_MAP = {
        "rsi_oversold": lambda r: float(r.get("RSI", 50)) < 30,
        "rsi_overbought": lambda r: float(r.get("RSI", 50)) > 70,
        "macd_cross_bull": lambda r: float(r.get("MACD", 0)) > float(r.get("MACD_signal", 0)) and float(r.get("MACD_hist", 0)) > 0 and float(r.get("MACD_hist", 0)) < abs(float(r.get("MACD", 1)) * 0.1),
        "macd_cross_bear": lambda r: float(r.get("MACD", 0)) < float(r.get("MACD_signal", 0)) and float(r.get("MACD_hist", 0)) < 0 and abs(float(r.get("MACD_hist", 0))) < abs(float(r.get("MACD", 1)) * 0.1),
        "volume_spike": lambda r: float(r.get("volume_ratio", 1)) > 2.5,
        "bollinger_squeeze": lambda r: float(r.get("bb_width", 0.5)) < 0.03,
        "strong_trend": lambda r: float(r.get("ADX", 0)) > 40 and float(r.get("efficiency_ratio", 0)) > 0.6,
        "ichimoku_bull": lambda r: float(r.get("ichimoku_cloud_pos", 0)) > 0 and float(r.get("ichimoku_tk", 0)) > 0,
        "ichimoku_bear": lambda r: float(r.get("ichimoku_cloud_pos", 0)) < 0 and float(r.get("ichimoku_tk", 0)) < 0,
    }

    check_fn = CONDITION_MAP.get(condition)
    if not check_fn:
        return []

    results = []
    interval = state.coin_manager.active_interval

    for symbol in candidates:
        if len(results) >= limit:
            break
        try:
            data = state.coin_manager.fetcher.fetch_with_cache(symbol, interval)
            enriched = indicators.compute_all(data.df.copy(), state.config.indicators)
            row = enriched.iloc[-1]
            row_dict = {k: (0 if (hasattr(v, '__float__') and (v != v)) else v) for k, v in row.items()}

            if check_fn(row_dict):
                ticker = next((t for t in tickers if t["symbol"] == symbol), {})
                results.append({
                    "symbol": symbol,
                    "baseAsset": ticker.get("baseAsset", symbol.replace("USDT", "")),
                    "lastPrice": ticker.get("lastPrice", 0),
                    "priceChangePercent": ticker.get("priceChangePercent", 0),
                    "rsi": round(float(row_dict.get("RSI", 0)), 1),
                    "adx": round(float(row_dict.get("ADX", 0)), 1),
                    "volume_ratio": round(float(row_dict.get("volume_ratio", 0)), 2),
                    "macd_bullish": bool(float(row_dict.get("MACD", 0)) > float(row_dict.get("MACD_signal", 0))),
                    "bb_width": round(float(row_dict.get("bb_width", 0)), 4),
                })
        except Exception:
            continue

    return results


def run_server(host: str = "0.0.0.0", port: int = 8000, config: Optional[AppConfig] = None):
    import uvicorn
    app = create_app(config)
    print(f"\n  BTCDump Web UI starting at http://localhost:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
