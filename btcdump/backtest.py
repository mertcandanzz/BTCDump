"""Walk-forward backtesting engine with threshold optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from btcdump import indicators
from btcdump.config import AppConfig
from btcdump.features import FeatureEngineer
from btcdump.models import ModelPipeline, TrainedEnsemble
from btcdump.signals import Signal, SignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest output."""

    total_signals: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    signal_accuracy: Dict[str, float]
    equity_curve: pd.DataFrame
    signals: List[Signal]
    optimal_thresholds: Dict[str, float]


class BacktestEngine:
    """Walk-forward backtester: no future data leakage.

    For each test point:
    1. Train model on data [0..t]
    2. Predict at t+1
    3. Compare prediction to actual at t+1
    4. Record signal and outcome
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._pipeline = ModelPipeline(config)

    def run(
        self,
        raw_df: pd.DataFrame,
        symbol: str = "",
        interval: str = "",
        retrain_every: int = 50,
        progress_callback: Optional[callable] = None,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            raw_df: Raw OHLCV DataFrame (no indicators).
            retrain_every: Retrain model every N candles for efficiency.
            progress_callback: Called with (current_step, total_steps).
        """
        min_train = self._config.model.min_train_size
        warmup = 60  # indicator warmup candles
        start_idx = min_train + warmup
        total_steps = len(raw_df) - 1 - start_idx

        if total_steps <= 0:
            return self._empty_result()

        current_ensemble: Optional[TrainedEnsemble] = None
        signal_gen = SignalGenerator(self._config.signal)
        results: List[Tuple[Signal, float]] = []

        for step in range(start_idx, len(raw_df) - 1):
            # Retrain periodically
            if current_ensemble is None or (step - start_idx) % retrain_every == 0:
                train_df = raw_df.iloc[:step].copy()
                try:
                    current_ensemble = self._pipeline.train_walk_forward(
                        train_df, symbol=symbol, interval=interval,
                    )
                except (ValueError, Exception) as exc:
                    logger.warning("Training failed at step %d: %s", step, exc)
                    continue

            # Predict using only data visible up to current step
            visible_df = raw_df.iloc[: step + 1].copy()
            try:
                prediction, model_conf, indiv_preds = self._pipeline.predict(
                    current_ensemble, visible_df,
                )
            except (ValueError, Exception):
                continue

            current_price = float(raw_df["close"].iloc[step])
            actual_next = float(raw_df["close"].iloc[step + 1])

            enriched = indicators.compute_all(visible_df, self._config.indicators)
            if enriched.dropna(subset=["RSI", "MACD"]).empty:
                continue

            indicator_row = enriched.iloc[-1]
            signal = signal_gen.generate(
                current_price, prediction, model_conf, indiv_preds, indicator_row,
            )

            actual_return = ((actual_next - current_price) / current_price) * 100.0
            results.append((signal, actual_return))

            if progress_callback:
                progress_callback(step - start_idx + 1, total_steps)

        return self._compile_results(results)

    # Trading fee (Binance spot: 0.1% per trade, entry + exit = 0.2% round trip)
    FEE_PER_TRADE_PCT = 0.1  # 0.1% per side

    def _compile_results(
        self, results: List[Tuple[Signal, float]],
    ) -> BacktestResult:
        """Compute performance metrics from (signal, actual_return) pairs.

        Includes realistic Binance trading fees (0.1% per side = 0.2% round trip).
        """
        if not results:
            return self._empty_result()

        signals = [r[0] for r in results]
        wins: List[float] = []
        losses: List[float] = []
        equity = [100.0]
        fee = self.FEE_PER_TRADE_PCT * 2  # round trip

        for signal, actual_ret in results:
            if signal.direction in ("BUY", "STRONG BUY"):
                pnl = actual_ret - fee  # subtract round-trip fee
            elif signal.direction in ("SELL", "STRONG SELL"):
                pnl = -actual_ret - fee  # subtract round-trip fee
            else:
                pnl = 0.0  # HOLD = no trade = no fee

            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(pnl)

            equity.append(equity[-1] * (1.0 + pnl / 100.0))

        equity_s = pd.Series(equity)
        cummax = equity_s.cummax()
        drawdown = ((equity_s - cummax) / cummax * 100.0).fillna(0.0)

        total_trades = len(wins) + len(losses)
        win_rate = len(wins) / total_trades if total_trades else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        # Sharpe ratio (annualized, assuming hourly candles)
        pnl_list = []
        for sig, ret in results:
            if sig.direction in ("BUY", "STRONG BUY"):
                pnl_list.append(ret)
            elif sig.direction in ("SELL", "STRONG SELL"):
                pnl_list.append(-ret)
        pnl_series = pd.Series(pnl_list) if pnl_list else pd.Series([0.0])
        sharpe = 0.0
        if len(pnl_series) > 1 and pnl_series.std() > 0:
            sharpe = float(pnl_series.mean() / pnl_series.std() * np.sqrt(252 * 24))

        optimal = self._optimize_thresholds(results)

        return BacktestResult(
            total_signals=len(results),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(float(drawdown.min()), 2),
            sharpe_ratio=round(sharpe, 2),
            total_return_pct=round((equity[-1] / equity[0] - 1.0) * 100.0, 2),
            avg_win_pct=round(float(np.mean(wins)), 4) if wins else 0.0,
            avg_loss_pct=round(float(np.mean(losses)), 4) if losses else 0.0,
            signal_accuracy=self._per_signal_accuracy(results),
            equity_curve=pd.DataFrame({
                "equity": equity,
                "drawdown": drawdown.values,
            }),
            signals=signals,
            optimal_thresholds=optimal,
        )

    def _optimize_thresholds(
        self, results: List[Tuple[Signal, float]],
    ) -> Dict[str, float]:
        """Grid search for thresholds that maximize profit factor."""
        best_pf = 0.0
        best: Dict[str, float] = {}

        for buy_t in np.arange(0.2, 2.01, 0.1):
            for sell_t in np.arange(-2.0, -0.19, 0.1):
                pf = self._evaluate_thresholds(results, float(buy_t), float(sell_t))
                if pf > best_pf:
                    best_pf = pf
                    best = {
                        "buy_threshold": round(float(buy_t), 1),
                        "sell_threshold": round(float(sell_t), 1),
                        "strong_buy_threshold": round(float(buy_t) * 2, 1),
                        "strong_sell_threshold": round(float(sell_t) * 2, 1),
                        "profit_factor": round(best_pf, 2),
                    }

        return best

    @staticmethod
    def _evaluate_thresholds(
        results: List[Tuple[Signal, float]], buy_t: float, sell_t: float,
    ) -> float:
        gross_profit = 0.0
        gross_loss = 0.0
        for signal, actual_ret in results:
            chg = signal.change_pct
            if chg > buy_t:
                pnl = actual_ret
            elif chg < sell_t:
                pnl = -actual_ret
            else:
                pnl = 0.0
            if pnl > 0:
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += abs(pnl)
        return gross_profit / (gross_loss + 1e-10)

    @staticmethod
    def _per_signal_accuracy(
        results: List[Tuple[Signal, float]],
    ) -> Dict[str, float]:
        by_type: Dict[str, Dict[str, int]] = {}
        for signal, actual_ret in results:
            d = signal.direction
            if d not in by_type:
                by_type[d] = {"correct": 0, "total": 0}
            by_type[d]["total"] += 1
            if d in ("BUY", "STRONG BUY") and actual_ret > 0:
                by_type[d]["correct"] += 1
            elif d in ("SELL", "STRONG SELL") and actual_ret < 0:
                by_type[d]["correct"] += 1
            elif d == "HOLD":
                by_type[d]["correct"] += 1
        return {
            k: round(v["correct"] / v["total"], 4)
            for k, v in by_type.items()
            if v["total"] > 0
        }

    @staticmethod
    def _empty_result() -> BacktestResult:
        return BacktestResult(
            total_signals=0, win_rate=0, profit_factor=0,
            max_drawdown_pct=0, sharpe_ratio=0, total_return_pct=0,
            avg_win_pct=0, avg_loss_pct=0, signal_accuracy={},
            equity_curve=pd.DataFrame({"equity": [100.0], "drawdown": [0.0]}),
            signals=[], optimal_thresholds={},
        )
