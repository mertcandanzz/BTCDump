"""Multi-coin state management, parallel signal computation, watchlist."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Callable, Dict, List, Optional

from btcdump import indicators
from btcdump.config import AppConfig
from btcdump.data import DataFetcher
from btcdump.models import ModelPipeline, TrainedEnsemble
from btcdump.signals import Signal, SignalGenerator

logger = logging.getLogger(__name__)


class CoinManager:
    """Manages multi-coin state and signal computation."""

    def __init__(
        self,
        fetcher: DataFetcher,
        pipeline: ModelPipeline,
        signal_gen: SignalGenerator,
        config: AppConfig,
    ) -> None:
        self.fetcher = fetcher
        self.pipeline = pipeline
        self.signal_gen = signal_gen
        self.config = config

        # Active coin (single analysis mode)
        self.active_symbol: str = config.data.default_symbol
        self.active_interval: str = config.data.default_interval
        self.active_ensemble: Optional[TrainedEnsemble] = None
        self.active_signal_data: Dict = {}

        # Watchlist (comparison mode)
        self.watchlist: List[str] = list(config.data.default_watchlist)

        # Caches
        self.signal_cache: Dict[str, Dict] = {}
        self.signal_cache_times: Dict[str, float] = {}
        self.ensembles: Dict[str, TrainedEnsemble] = {}

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.data.signal_workers,
            thread_name_prefix="signal",
        )

        # Try loading active ensemble
        self.active_ensemble = pipeline.load(self.active_symbol, self.active_interval)
        if self.active_ensemble:
            self.ensembles[self.active_symbol] = self.active_ensemble

    # ── Coin Discovery ────────────────────────────────────

    def get_coins(self, query: str = "", limit: int = 100) -> List[Dict]:
        """Return available USDT pairs with 24h stats, optionally filtered."""
        try:
            tickers = self.fetcher.fetch_tickers()
        except Exception:
            logger.exception("Failed to fetch tickers")
            return []

        if query:
            q = query.upper()
            tickers = [t for t in tickers if q in t["symbol"] or q in t["baseAsset"]]

        return tickers[:limit]

    # ── Active Coin ───────────────────────────────────────

    def set_active_coin(self, symbol: str) -> None:
        """Switch active coin for single analysis mode."""
        self.active_symbol = symbol
        # Load cached ensemble if available
        self.active_ensemble = self.ensembles.get(symbol)
        if not self.active_ensemble:
            self.active_ensemble = self.pipeline.load(symbol, self.active_interval)
            if self.active_ensemble:
                self.ensembles[symbol] = self.active_ensemble
        # Load cached signal if available
        self.active_signal_data = self.signal_cache.get(symbol, {})
        logger.info("Active coin set to %s (model: %s)", symbol, "loaded" if self.active_ensemble else "none")

    def set_interval(self, interval: str) -> None:
        """Change interval. Invalidates all ensembles."""
        self.active_interval = interval
        self.ensembles.clear()
        self.signal_cache.clear()
        self.signal_cache_times.clear()
        self.active_ensemble = self.pipeline.load(self.active_symbol, interval)
        if self.active_ensemble:
            self.ensembles[self.active_symbol] = self.active_ensemble

    # ── Watchlist ─────────────────────────────────────────

    def set_watchlist(self, symbols: List[str]) -> List[str]:
        """Set watchlist (max 15). Returns validated list."""
        try:
            valid_symbols = {t["symbol"] for t in self.fetcher.fetch_tickers()}
        except Exception:
            valid_symbols = set()

        validated = []
        for s in symbols[:15]:
            s = s.upper()
            if not valid_symbols or s in valid_symbols:
                validated.append(s)

        self.watchlist = validated
        logger.info("Watchlist updated: %s", validated)
        return validated

    # ── Multi-Timeframe ─────────────────────────────────

    MULTI_TF = ["15m", "1h", "4h", "1d"]

    def compute_multi_tf_signal(self, symbol: str) -> Dict:
        """Compute signals across 4 timeframes for alignment analysis."""
        results = {}
        for tf in self.MULTI_TF:
            try:
                data = self.fetcher.fetch_with_cache(symbol, tf)
                ensemble = self.pipeline.load(symbol, tf)
                if not ensemble:
                    ensemble = self.pipeline.train_walk_forward(
                        data.df, symbol=symbol, interval=tf,
                    )
                    self.pipeline.save(ensemble)
                pred, conf, indiv = self.pipeline.predict(ensemble, data.df)
                enriched = indicators.compute_all(data.df.copy(), self.config.indicators)
                current = float(data.df["close"].iloc[-1])
                sig = self.signal_gen.generate(current, pred, conf, indiv, enriched.iloc[-1])
                results[tf] = {
                    "direction": sig.direction,
                    "confidence": round(sig.confidence, 1),
                    "change_pct": round(sig.change_pct, 2),
                    "rsi": round(float(enriched.iloc[-1].get("RSI", 0)), 1),
                }
            except Exception as e:
                results[tf] = {"direction": "ERROR", "confidence": 0, "error": str(e)}

        directions = [r.get("direction", "") for r in results.values()]
        bullish = sum(1 for d in directions if "BUY" in d)
        bearish = sum(1 for d in directions if "SELL" in d)
        active = len([d for d in directions if d not in ("HOLD", "ERROR", "")])

        if active == 0:
            alignment, alignment_pct = "neutral", 0
        elif bullish > bearish:
            alignment, alignment_pct = "bullish", round(bullish / len(self.MULTI_TF) * 100)
        elif bearish > bullish:
            alignment, alignment_pct = "bearish", round(bearish / len(self.MULTI_TF) * 100)
        else:
            alignment, alignment_pct = "mixed", 50

        return {"timeframes": results, "alignment": alignment, "alignment_pct": alignment_pct}

    # ── Signal Computation ────────────────────────────────

    def compute_signal(self, symbol: str) -> Dict:
        """Compute full signal for one coin. Sync, CPU-bound."""
        try:
            data = self.fetcher.fetch_with_cache(symbol, self.active_interval)

            # Load or train ensemble
            ensemble = self.ensembles.get(symbol)
            if not ensemble:
                ensemble = self.pipeline.load(symbol, self.active_interval)

            if not ensemble or self.pipeline.should_retrain(ensemble, data.num_candles):
                ensemble = self.pipeline.train_walk_forward(
                    data.df, symbol=symbol, interval=self.active_interval,
                )
                self.pipeline.save(ensemble)

            self.ensembles[symbol] = ensemble

            # Predict
            pred, conf, indiv = self.pipeline.predict(ensemble, data.df)
            enriched = indicators.compute_all(data.df.copy(), self.config.indicators)
            current_price = float(data.df["close"].iloc[-1])
            row = enriched.iloc[-1]

            signal = self.signal_gen.generate(
                current_price, pred, conf, indiv, row,
            )

            signal_data = self._signal_to_dict(signal, ensemble, row, symbol)

            # Update caches
            self.signal_cache[symbol] = signal_data
            self.signal_cache_times[symbol] = time.time()

            if symbol == self.active_symbol:
                self.active_signal_data = signal_data
                self.active_ensemble = ensemble

            return signal_data

        except Exception as exc:
            logger.exception("Signal computation failed for %s", symbol)
            return {"symbol": symbol, "error": str(exc), "status": "error"}

    def refresh_active_signal(self) -> Dict:
        """Refresh signal for the active coin."""
        return self.compute_signal(self.active_symbol)

    def compute_watchlist_signals(
        self, on_progress: Optional[Callable] = None,
    ) -> Dict[str, Dict]:
        """Compute signals for all watchlist coins in parallel."""
        results: Dict[str, Dict] = {}
        to_compute: List[str] = []

        for symbol in self.watchlist:
            if self._is_cache_fresh(symbol):
                results[symbol] = self.signal_cache[symbol]
            else:
                to_compute.append(symbol)

        total = len(to_compute)
        completed = 0

        futures = {
            self._executor.submit(self.compute_signal, sym): sym
            for sym in to_compute
        }

        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                results[symbol] = future.result(timeout=120)
            except Exception as exc:
                results[symbol] = {"symbol": symbol, "error": str(exc), "status": "error"}
            completed += 1
            if on_progress:
                on_progress(completed, total, symbol)

        return results

    # ── Mini Chart ────────────────────────────────────────

    def get_mini_chart_data(self, symbol: str) -> List[float]:
        """Return close prices for sparkline."""
        try:
            return self.fetcher.fetch_mini_chart(symbol, self.active_interval)
        except Exception:
            logger.exception("Mini chart fetch failed for %s", symbol)
            return []

    # ── Overview ──────────────────────────────────────────

    def get_watchlist_overview(self) -> List[Dict]:
        """Full overview: ticker data + ALL cached signal fields + mini chart."""
        try:
            tickers = {t["symbol"]: t for t in self.fetcher.fetch_tickers()}
        except Exception:
            tickers = {}

        overview = []
        for symbol in self.watchlist:
            ticker = tickers.get(symbol, {})
            cached = self.signal_cache.get(symbol)

            entry = {
                "symbol": symbol,
                "baseAsset": ticker.get("baseAsset", symbol.replace("USDT", "")),
                "lastPrice": ticker.get("lastPrice", 0),
                "priceChangePercent": ticker.get("priceChangePercent", 0),
                "quoteVolume": ticker.get("quoteVolume", 0),
            }

            if cached and "error" not in cached:
                # Include ALL signal fields for detailed compare
                entry.update({
                    "direction": cached.get("direction", ""),
                    "confidence": cached.get("confidence", 0),
                    "change_pct": cached.get("change_pct", 0),
                    "predicted_price": cached.get("predicted_price", 0),
                    "rsi": cached.get("rsi", 0),
                    "macd_bullish": cached.get("macd_bullish", False),
                    "macd_val": cached.get("macd_val", 0),
                    "stoch_k": cached.get("stoch_k", 0),
                    "adx": cached.get("adx", 0),
                    "atr": cached.get("atr", 0),
                    "volume_ratio": cached.get("volume_ratio", 0),
                    "risk_reward": cached.get("risk_reward", 0),
                    "model_agreement": cached.get("model_agreement", 0),
                    "mape": cached.get("mape", 0),
                    "reasons": cached.get("reasons", []),
                    "indicator_confluence": cached.get("indicator_confluence", 0),
                    "weights": cached.get("weights", {}),
                    "status": "ready",
                })
            else:
                entry["status"] = "pending"

            try:
                entry["mini_chart"] = self.fetcher.fetch_mini_chart(
                    symbol, self.active_interval,
                )
            except Exception:
                entry["mini_chart"] = []

            overview.append(entry)

        return overview

    def get_compare_context(self) -> str:
        """Build a summary string of watchlist for AI chat context."""
        lines = [f"WATCHLIST COMPARISON ({self.active_interval} timeframe):"]
        for symbol in self.watchlist:
            cached = self.signal_cache.get(symbol, {})
            if cached and "error" not in cached:
                lines.append(
                    f"  {symbol}: {cached.get('direction','?')} "
                    f"(conf={cached.get('confidence',0):.0f}%, "
                    f"RSI={cached.get('rsi','?')}, "
                    f"chg={cached.get('change_pct',0):+.2f}%)"
                )
            else:
                lines.append(f"  {symbol}: signal pending")
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────

    def _is_cache_fresh(self, symbol: str) -> bool:
        t = self.signal_cache_times.get(symbol, 0)
        return (time.time() - t) < self.config.data.cache_ttl_seconds and symbol in self.signal_cache

    @staticmethod
    def _signal_to_dict(
        sig: Signal, ensemble: TrainedEnsemble, row, symbol: str,
    ) -> Dict:
        import pandas as pd

        def _safe(val, default=0):
            if pd.isna(val):
                return default
            return val

        return {
            "symbol": symbol,
            "current_price": sig.current_price,
            "predicted_price": sig.predicted_price,
            "change_pct": sig.change_pct,
            "direction": sig.direction,
            "confidence": sig.confidence,
            "model_agreement": sig.model_agreement,
            "indicator_confluence": sig.indicator_confluence,
            "risk_reward": sig.risk_reward,
            "reasons": sig.reasons,
            "rsi": round(float(_safe(row.get("RSI", 0))), 1),
            "macd_bullish": bool(_safe(row.get("MACD", 0)) > _safe(row.get("MACD_signal", 0))),
            "macd_val": round(float(_safe(row.get("MACD", 0))), 2),
            "stoch_k": round(float(_safe(row.get("stoch_k", 0))), 1),
            "adx": round(float(_safe(row.get("ADX", 0))), 1),
            "atr": round(float(_safe(row.get("ATR", 0))), 2),
            "volume_ratio": round(float(_safe(row.get("volume_ratio", 0))), 2),
            "bb_upper": round(float(_safe(row.get("BB_upper", 0))), 2),
            "bb_lower": round(float(_safe(row.get("BB_lower", 0))), 2),
            "ma20": round(float(_safe(row.get("ma20", 0))), 2),
            "mape": round(ensemble.avg_mape * 100, 2),
            "weights": {k: round(v, 3) for k, v in ensemble.weights.items()},
            "interval": ensemble.interval,
            "status": "ready",
        }
