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
        "efficiency_ratio":"Kaufman ER - trend efficiency (0=chop,1=clean trend)","choppiness":"Choppiness Index (>61.8=chop,<38.2=trend)","adx_slope":"ADX 5-bar change (trend strengthening/weakening)","tsi":"True Strength Index (double-smoothed momentum)","rsi_divergence":"Price-RSI divergence (positive=bullish div)","returns_10":"10-candle return","returns_20":"20-candle return","momentum_quality":"Directional Sharpe (return/volatility)","garch_proxy":"GARCH proxy (short/long vol ratio, >1=expanding)","vol_of_vol":"Volatility of volatility (regime transition signal)","yang_zhang_vol":"Yang-Zhang vol (state-of-art OHLCV estimator)","volume_trend":"Volume slope (rising=accumulation)","amihud_illiq":"Amihud illiquidity (higher=less liquid)","hour_sin":"Hour sine (cyclical time feature)","hour_cos":"Hour cosine (cyclical time feature)","dow_sin":"Day-of-week sine (weekend/weekday cycle)","dow_cos":"Day-of-week cosine","skewness_20":"Return skewness (neg=left tail risk)","kurtosis_20":"Return kurtosis (high=extreme moves likely)","keltner_position":"Keltner Channel position (ATR-based bands)","squeeze_ratio":"BB/Keltner squeeze (<1=squeeze,breakout imminent)","engulfing_score":"Engulfing pattern strength (pos=bullish)","consecutive_dir":"Consecutive candle direction streak","ofi_14":"Order Flow Imbalance 14-bar (buy vs sell pressure proxy)","pv_divergence":"Price-Volume divergence (high=thin move, low=confirmed)","ichimoku_tk":"Ichimoku Tenkan-Kijun cross (% of price, pos=bullish)","ichimoku_cloud_pos":"Price position vs Ichimoku cloud (>0=above)","ichimoku_cloud_width":"Ichimoku cloud thickness (% of price, thin=weak S/R)","ichimoku_chikou":"Chikou span: price vs 26 bars ago (% change)","ichimoku_kijun_dist":"Distance from Kijun-sen baseline (mean-reversion)","vp_poc_dist":"Distance from Volume POC (Point of Control, %)","vp_va_position":"Position within Value Area (0=low, 1=high, >1=above)","pivot_dist":"Distance from pivot point (% of price)","pivot_r1_dist":"Distance to R1 resistance (%)","pivot_s1_dist":"Distance to S1 support (%)","pivot_position":"Position within S2-R2 range (0=S2, 1=R2)","pattern_doji":"Doji strength (1=perfect doji, indecision)","pattern_hammer":"Hammer/HangingMan strength (bullish reversal)","pattern_shooting_star":"Shooting Star strength (bearish reversal)","pattern_three_soldiers":"Three White Soldiers (strong bull continuation)","pattern_three_crows":"Three Black Crows (strong bear continuation)","pattern_morning_star":"Morning Star (bullish reversal, 3-candle)","pattern_evening_star":"Evening Star (bearish reversal, 3-candle)","trade_intensity":"Volume per unit price move z-score (high=absorption)","pin_bar_score":"Pin bar strength (pos=bullish, neg=bearish reversal)","gap_pct":"Gap % from prev close to open (crypto overnight gaps)","intrabar_vol_ratio":"Total range / body size (high=choppy intrabar)","close_position_avg":"5-bar avg close position in range (persistent pressure)","whale_score":"Whale activity score (high vol + small body = absorption)","smart_money_div":"Smart money divergence (1=bullish, -1=bearish accumulation/distribution)","price_entropy":"Shannon entropy of returns (high=uncertain/random, low=predictable)","hurst_exponent":"Hurst exponent (>0.5=trending, <0.5=mean-reverting, 0.5=random)","autocorr_1":"Return auto-correlation lag-1 (pos=momentum, neg=reversal)","autocorr_5":"Return auto-correlation lag-5 (weekly pattern on 1h data)","di_ratio":"DI+/DI- ratio (>1=bullish dominance, <1=bearish)","di_spread":"DI+/DI- normalized spread (-1 to +1)","variance_ratio":"Lo-MacKinlay variance ratio (>1=trending, <1=mean-reverting)",
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
