"""Chart rendering: candlestick, multi-panel, equity curve."""

from __future__ import annotations

import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcdump.signals import Signal

logger = logging.getLogger(__name__)

# Try mplfinance for candlestick charts; fall back to line charts
try:
    import mplfinance as mpf

    _HAS_MPF = True
except ImportError:
    _HAS_MPF = False
    logger.info("mplfinance not installed; using line charts as fallback")


class ChartRenderer:
    """Professional multi-panel chart renderer."""

    def __init__(self) -> None:
        plt.style.use("dark_background")

    def price_chart(
        self,
        df: pd.DataFrame,
        signals: Optional[List[Signal]] = None,
        prediction: Optional[float] = None,
    ) -> None:
        """Render price chart with indicators.

        Uses candlestick (mplfinance) if available, else line chart.
        Includes RSI sub-panel and volume.
        """
        if _HAS_MPF:
            self._candlestick_chart(df, signals, prediction)
        else:
            self._line_chart(df, signals, prediction)

    def equity_curve(self, equity_df: pd.DataFrame) -> None:
        """2-panel chart: equity curve + drawdown."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]},
        )

        ax1.plot(equity_df["equity"], linewidth=2, color="#00d4aa")
        ax1.set_title("Equity Curve", fontsize=14)
        ax1.set_ylabel("Equity ($)")
        ax1.grid(alpha=0.3)

        ax2.fill_between(
            equity_df.index, equity_df["drawdown"],
            alpha=0.6, color="#ff4444",
        )
        ax2.set_title("Drawdown", fontsize=12)
        ax2.set_ylabel("Drawdown %")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    # --- Private ---

    def _candlestick_chart(
        self,
        df: pd.DataFrame,
        signals: Optional[List[Signal]],
        prediction: Optional[float],
    ) -> None:
        """Render candlestick chart with mplfinance."""
        chart_df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        chart_df = chart_df.set_index("time")
        chart_df.index.name = "Date"

        # Build add-plots
        add_plots = []

        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            bb_upper = df.set_index("time")["BB_upper"]
            bb_lower = df.set_index("time")["BB_lower"]
            add_plots.append(mpf.make_addplot(bb_upper, color="gray", linestyle="--", width=0.7))
            add_plots.append(mpf.make_addplot(bb_lower, color="gray", linestyle="--", width=0.7))

        if "ma20" in df.columns:
            ma20 = df.set_index("time")["ma20"]
            add_plots.append(mpf.make_addplot(ma20, color="#ffaa00", width=1))

        if "RSI" in df.columns:
            rsi = df.set_index("time")["RSI"]
            add_plots.append(mpf.make_addplot(rsi, panel=2, color="#00d4aa", ylabel="RSI"))

        if "MACD" in df.columns and "MACD_signal" in df.columns:
            macd_line = df.set_index("time")["MACD"]
            macd_sig = df.set_index("time")["MACD_signal"]
            add_plots.append(mpf.make_addplot(macd_line, panel=3, color="#00aaff", ylabel="MACD"))
            add_plots.append(mpf.make_addplot(macd_sig, panel=3, color="#ff6600"))

        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            rc={"font.size": 9},
        )

        mpf.plot(
            chart_df,
            type="candle",
            style=style,
            volume=True,
            addplot=add_plots if add_plots else None,
            title="BTC/USDT",
            figsize=(16, 12),
            panel_ratios=(4, 1, 1, 1) if add_plots else (4, 1),
        )

    def _line_chart(
        self,
        df: pd.DataFrame,
        signals: Optional[List[Signal]],
        prediction: Optional[float],
    ) -> None:
        """Fallback line chart when mplfinance is not available."""
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 12),
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
        ax_price, ax_rsi, ax_vol = axes

        # Price + MA + BB
        ax_price.plot(df["time"], df["close"], label="BTC Price", linewidth=2, color="#00d4aa")

        if "ma20" in df.columns:
            ax_price.plot(df["time"], df["ma20"], label="MA20", linewidth=1, color="#ffaa00")
        if "BB_upper" in df.columns:
            ax_price.plot(df["time"], df["BB_upper"], "--", color="gray", linewidth=0.7)
            ax_price.plot(df["time"], df["BB_lower"], "--", color="gray", linewidth=0.7)

        if prediction is not None:
            ax_price.axhline(prediction, color="#ff6600", linestyle="--", label=f"Prediction: ${prediction:,.0f}")

        ax_price.legend()
        ax_price.set_title("BTC/USDT", fontsize=14)
        ax_price.grid(alpha=0.3)

        # RSI
        if "RSI" in df.columns:
            ax_rsi.plot(df["time"], df["RSI"], color="#00d4aa", linewidth=1)
            ax_rsi.axhline(70, linestyle="--", color="red", alpha=0.5)
            ax_rsi.axhline(30, linestyle="--", color="green", alpha=0.5)
            ax_rsi.set_ylabel("RSI")
            ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(alpha=0.3)

        # Volume
        colors = ["#00d4aa" if c >= o else "#ff4444"
                  for c, o in zip(df["close"], df["open"])]
        ax_vol.bar(df["time"], df["volume"], color=colors, alpha=0.6, width=0.8)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()
