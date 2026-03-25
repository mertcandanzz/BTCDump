"""CLI application: menu system, display logic, wiring all modules together."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from btcdump import indicators
from btcdump.backtest import BacktestEngine, BacktestResult
from btcdump.config import AppConfig
from btcdump.data import CandleData, DataFetcher
from btcdump.models import ModelPipeline, TrainedEnsemble
from btcdump.signals import Signal, SignalGenerator
from btcdump.utils import ensure_dirs, setup_logging
from btcdump.visualization import ChartRenderer

logger = logging.getLogger(__name__)

# ANSI colours
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

BANNER = r"""
{cyan}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ████████╗ ██████╗██████╗ ██╗   ██╗███╗   ███╗██████╗  ║
║   ██╔══██╗╚══██╔══╝██╔════╝██╔══██╗██║   ██║████╗ ████║██╔══██╗ ║
║   ██████╔╝   ██║   ██║     ██║  ██║██║   ██║██╔████╔██║██████╔╝ ║
║   ██╔══██╗   ██║   ██║     ██║  ██║██║   ██║██║╚██╔╝██║██╔═══╝  ║
║   ██████╔╝   ██║   ╚██████╗██████╔╝╚██████╔╝██║ ╚═╝ ██║██║      ║
║   ╚═════╝    ╚═╝    ╚═════╝╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═╝      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{reset}
""".format(cyan=CYAN, reset=RESET)

TIMEFRAMES = {
    "1": ("5m", "5 Minutes"),
    "2": ("30m", "30 Minutes"),
    "3": ("1h", "1 Hour"),
    "4": ("4h", "4 Hours"),
    "5": ("1d", "1 Day"),
}


class BTCDumpApp:
    """Main CLI application."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or AppConfig()
        self._fetcher = DataFetcher(self._config.data)
        self._pipeline = ModelPipeline(self._config)
        self._signal_gen = SignalGenerator(self._config.signal)
        self._backtest = BacktestEngine(self._config)
        self._chart = ChartRenderer()

        self._interval = self._config.data.default_interval
        self._symbol = self._config.data.default_symbol
        self._ensemble: Optional[TrainedEnsemble] = None
        self._last_signal: Optional[Signal] = None
        self._last_data: Optional[CandleData] = None

        ensure_dirs(self._config.data.cache_dir, self._config.model.models_dir)
        setup_logging(self._config.log_level, self._config.log_file)

    def run(self) -> None:
        """Main menu loop."""
        # Try loading persisted model
        self._ensemble = self._pipeline.load(self._symbol, self._interval)
        if self._ensemble:
            logger.info("Loaded saved model for %s/%s", self._symbol, self._interval)

        while True:
            self._show_banner()
            self._show_menu()
            choice = input(f"\n{CYAN}>> {RESET}").strip()
            self._handle_choice(choice)

    # --- Menu ---

    def _show_banner(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        print(BANNER)
        status = f"{WHITE}BTCDump v3.0 - Professional Signal Engine{RESET}"
        model_status = f"{GREEN}Model loaded{RESET}" if self._ensemble else f"{DIM}No model{RESET}"
        print(f"  {status}  |  {model_status}")
        print(f"  {DIM}Symbol: {self._symbol}  |  Interval: {self._interval}{RESET}")
        print(f"  {DIM}{'-' * 56}{RESET}")

    def _show_menu(self) -> None:
        print(f"""
  {WHITE}1.{RESET} Select Timeframe     {DIM}(current: {self._interval}){RESET}
  {WHITE}2.{RESET} Train & Predict
  {WHITE}3.{RESET} Show Live Chart
  {WHITE}4.{RESET} Run Backtest
  {WHITE}5.{RESET} Show Last Signal
  {WHITE}6.{RESET} Signal History
  {WHITE}7.{RESET} Auto Live Mode
  {WHITE}0.{RESET} Exit""")

    def _handle_choice(self, choice: str) -> None:
        handlers = {
            "0": self._exit,
            "1": self._select_timeframe,
            "2": self._train_and_predict,
            "3": self._show_chart,
            "4": self._run_backtest,
            "5": self._show_last_signal,
            "6": self._show_history,
            "7": self._auto_live,
        }
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print(f"\n{RED}Invalid choice{RESET}")
            input(f"\n{DIM}Press Enter...{RESET}")

    # --- Handlers ---

    def _exit(self) -> None:
        print(f"\n{CYAN}Goodbye!{RESET}")
        raise SystemExit(0)

    def _select_timeframe(self) -> None:
        print(f"\n{WHITE}Available Timeframes:{RESET}")
        for k, (code, label) in TIMEFRAMES.items():
            marker = f" {GREEN}<-{RESET}" if code == self._interval else ""
            print(f"  {k}. {label} ({code}){marker}")

        sel = input(f"\n{CYAN}Select: {RESET}").strip()
        if sel in TIMEFRAMES:
            self._interval = TIMEFRAMES[sel][0]
            # Invalidate cached model for different interval
            self._ensemble = self._pipeline.load(self._symbol, self._interval)
            print(f"{GREEN}Timeframe set to {TIMEFRAMES[sel][1]}{RESET}")
        else:
            print(f"{RED}Invalid selection{RESET}")
        input(f"\n{DIM}Press Enter...{RESET}")

    def _train_and_predict(self) -> None:
        print(f"\n{CYAN}Fetching market data...{RESET}")
        try:
            self._last_data = self._fetcher.fetch_with_cache(
                self._symbol, self._interval,
            )
        except Exception as exc:
            print(f"{RED}Failed to fetch data: {exc}{RESET}")
            input(f"\n{DIM}Press Enter...{RESET}")
            return

        if self._last_data.num_candles < 100:
            print(f"{RED}Insufficient data ({self._last_data.num_candles} candles){RESET}")
            input(f"\n{DIM}Press Enter...{RESET}")
            return

        print(f"{CYAN}Training ensemble with walk-forward validation...{RESET}")

        def on_fold(fold: int, total: int) -> None:
            print(f"  {DIM}Fold {fold}/{total} complete{RESET}")

        try:
            self._ensemble = self._pipeline.train_walk_forward(
                self._last_data.df,
                symbol=self._symbol,
                interval=self._interval,
                progress_callback=on_fold,
            )
        except ValueError as exc:
            print(f"{RED}Training failed: {exc}{RESET}")
            input(f"\n{DIM}Press Enter...{RESET}")
            return

        self._pipeline.save(self._ensemble)

        # Predict
        prediction, confidence, indiv = self._pipeline.predict(
            self._ensemble, self._last_data.df,
        )

        enriched = indicators.compute_all(
            self._last_data.df.copy(), self._config.indicators,
        )
        current_price = float(self._last_data.df["close"].iloc[-1])

        self._last_signal = self._signal_gen.generate(
            current_price, prediction, confidence, indiv, enriched.iloc[-1],
        )

        self._display_full_analysis(self._last_signal, self._ensemble)
        input(f"\n{DIM}Press Enter to continue...{RESET}")

    def _show_chart(self) -> None:
        print(f"\n{CYAN}Fetching data for chart...{RESET}")
        try:
            data = self._fetcher.fetch_with_cache(self._symbol, self._interval)
        except Exception as exc:
            print(f"{RED}Failed to fetch data: {exc}{RESET}")
            input(f"\n{DIM}Press Enter...{RESET}")
            return

        enriched = indicators.compute_all(data.df, self._config.indicators)
        prediction = self._last_signal.predicted_price if self._last_signal else None
        self._chart.price_chart(enriched, prediction=prediction)

    def _run_backtest(self) -> None:
        print(f"\n{CYAN}Fetching historical data...{RESET}")
        try:
            data = self._fetcher.fetch(self._symbol, self._interval)
        except Exception as exc:
            print(f"{RED}Failed to fetch data: {exc}{RESET}")
            input(f"\n{DIM}Press Enter...{RESET}")
            return

        total_est = len(data.df) - self._config.model.min_train_size - 60
        print(f"{CYAN}Running walk-forward backtest ({total_est} steps)...{RESET}")
        print(f"{DIM}This may take several minutes.{RESET}\n")

        last_pct = 0

        def on_progress(step: int, total: int) -> None:
            nonlocal last_pct
            pct = int(step / total * 100) if total > 0 else 0
            if pct >= last_pct + 10:
                last_pct = pct
                print(f"  {DIM}Progress: {pct}%{RESET}")

        result = self._backtest.run(
            data.df,
            symbol=self._symbol,
            interval=self._interval,
            progress_callback=on_progress,
        )

        self._display_backtest(result)

        if result.total_signals > 0:
            show = input(f"\n{CYAN}Show equity curve? (y/n): {RESET}").strip().lower()
            if show == "y":
                self._chart.equity_curve(result.equity_curve)

            if result.optimal_thresholds:
                print(f"\n{YELLOW}Optimal thresholds found:{RESET}")
                for k, v in result.optimal_thresholds.items():
                    print(f"  {k}: {v}")
                apply_ = input(f"\n{CYAN}Apply optimal thresholds? (y/n): {RESET}").strip().lower()
                if apply_ == "y":
                    self._signal_gen.update_thresholds(
                        buy=result.optimal_thresholds["buy_threshold"],
                        sell=result.optimal_thresholds["sell_threshold"],
                        strong_buy=result.optimal_thresholds["strong_buy_threshold"],
                        strong_sell=result.optimal_thresholds["strong_sell_threshold"],
                    )
                    print(f"{GREEN}Thresholds updated!{RESET}")

        input(f"\n{DIM}Press Enter...{RESET}")

    def _show_last_signal(self) -> None:
        if self._last_signal:
            self._display_signal(self._last_signal)
        else:
            print(f"\n{DIM}No signal generated yet. Run Train & Predict first.{RESET}")
        input(f"\n{DIM}Press Enter...{RESET}")

    def _show_history(self) -> None:
        history = self._signal_gen.history
        if not history:
            print(f"\n{DIM}No signal history.{RESET}")
            input(f"\n{DIM}Press Enter...{RESET}")
            return

        print(f"\n{WHITE}Signal History ({len(history)} signals){RESET}")
        print(f"{'Time':<20} {'Signal':<12} {'Conf':<8} {'Price':<14} {'Pred':<14} {'Chg%':<8}")
        print("-" * 76)

        for sig in history[-20:]:  # last 20
            color = self._signal_color(sig.direction)
            print(
                f"{sig.timestamp.strftime('%H:%M:%S'):<20} "
                f"{color}{sig.direction:<12}{RESET} "
                f"{sig.confidence:<8.1f} "
                f"${sig.current_price:<13,.2f} "
                f"${sig.predicted_price:<13,.2f} "
                f"{sig.change_pct:+.2f}%"
            )

        input(f"\n{DIM}Press Enter...{RESET}")

    def _auto_live(self) -> None:
        try:
            delay = int(input(f"\n{CYAN}Refresh interval (seconds, default 60): {RESET}").strip() or "60")
        except ValueError:
            delay = 60

        print(f"\n{CYAN}Auto Live Mode started (Ctrl+C to stop){RESET}\n")

        while True:
            try:
                data = self._fetcher.fetch(self._symbol, self._interval)

                # Retrain only when needed
                if self._ensemble is None or self._pipeline.should_retrain(
                    self._ensemble, data.num_candles,
                ):
                    print(f"{YELLOW}Retraining models...{RESET}")
                    self._ensemble = self._pipeline.train_walk_forward(
                        data.df, symbol=self._symbol, interval=self._interval,
                    )
                    self._pipeline.save(self._ensemble)

                prediction, confidence, indiv = self._pipeline.predict(
                    self._ensemble, data.df,
                )

                enriched = indicators.compute_all(data.df, self._config.indicators)
                current_price = float(data.df["close"].iloc[-1])

                self._last_signal = self._signal_gen.generate(
                    current_price, prediction, confidence, indiv, enriched.iloc[-1],
                )

                self._show_banner()
                print(f"\n  {CYAN}AUTO LIVE MODE{RESET}  |  Refresh: {delay}s  |  {DIM}Ctrl+C to stop{RESET}\n")
                self._display_signal(self._last_signal)

                time.sleep(delay)

            except KeyboardInterrupt:
                print(f"\n{YELLOW}Auto Live Mode stopped.{RESET}")
                input(f"\n{DIM}Press Enter...{RESET}")
                break
            except Exception as exc:
                logger.exception("Auto-live error")
                print(f"{RED}Error: {exc}{RESET}")
                time.sleep(delay)

    # --- Display helpers ---

    def _display_full_analysis(self, sig: Signal, ensemble: TrainedEnsemble) -> None:
        color = self._signal_color(sig.direction)
        print(f"\n{'=' * 60}")
        print(f"  {BOLD}BTC ANALYSIS{RESET}")
        print(f"{'=' * 60}")
        print(f"  Current Price:    ${sig.current_price:,.2f}")
        print(f"  AI Prediction:    ${sig.predicted_price:,.2f}")
        print(f"  Change:           {sig.change_pct:+.2f}%")
        print(f"  Signal:           {color}{BOLD}{sig.direction}{RESET}")
        print(f"  Confidence:       {sig.confidence:.1f}%")
        print(f"  Model Agreement:  {sig.model_agreement:.0%}")
        print(f"  Indicators:       {sig.indicator_confluence}/5 confirming")
        print(f"  Risk/Reward:      {sig.risk_reward:.2f}")
        print(f"  Model MAPE:       {ensemble.avg_mape * 100:.2f}%")
        print(f"  Ensemble Weights: " + ", ".join(
            f"{n}={w:.1%}" for n, w in ensemble.weights.items()
        ))
        if sig.reasons:
            print(f"  Reasons:")
            for reason in sig.reasons:
                print(f"    {DIM}- {reason}{RESET}")
        print(f"{'=' * 60}")

    def _display_signal(self, sig: Signal) -> None:
        color = self._signal_color(sig.direction)
        print(f"\n  {WHITE}Price:{RESET}      ${sig.current_price:,.2f}")
        print(f"  {WHITE}Prediction:{RESET} ${sig.predicted_price:,.2f} ({sig.change_pct:+.2f}%)")
        print(f"  {WHITE}Signal:{RESET}     {color}{BOLD}{sig.direction}{RESET}")
        print(f"  {WHITE}Confidence:{RESET} {sig.confidence:.1f}%")
        print(f"  {WHITE}R/R:{RESET}        {sig.risk_reward:.2f}")
        print(f"  {DIM}{sig.timestamp.strftime('%Y-%m-%d %H:%M:%S')}{RESET}")

    def _display_backtest(self, result: BacktestResult) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {BOLD}BACKTEST RESULTS{RESET}")
        print(f"{'=' * 60}")
        print(f"  Total Signals:    {result.total_signals}")
        print(f"  Win Rate:         {result.win_rate:.1%}")
        print(f"  Profit Factor:    {result.profit_factor:.2f}")
        print(f"  Max Drawdown:     {result.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"  Total Return:     {result.total_return_pct:+.2f}%")
        print(f"  Avg Win:          {result.avg_win_pct:+.4f}%")
        print(f"  Avg Loss:         {result.avg_loss_pct:+.4f}%")

        if result.signal_accuracy:
            print(f"\n  {WHITE}Per-Signal Accuracy:{RESET}")
            for sig_type, acc in sorted(result.signal_accuracy.items()):
                print(f"    {sig_type:<14} {acc:.1%}")

        print(f"{'=' * 60}")

    @staticmethod
    def _signal_color(direction: str) -> str:
        if direction in ("BUY", "STRONG BUY"):
            return GREEN
        if direction in ("SELL", "STRONG SELL"):
            return RED
        return YELLOW
