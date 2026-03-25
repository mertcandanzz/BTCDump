"""Signal generation with confidence scoring and indicator confluence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from btcdump.config import SignalConfig

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A single trading signal with full context."""

    direction: str              # STRONG BUY, BUY, HOLD, SELL, STRONG SELL
    confidence: float           # 0-100%
    predicted_price: float
    current_price: float
    change_pct: float
    model_agreement: float      # 0-1
    indicator_confluence: int   # count of confirming indicators (out of 5)
    risk_reward: float
    timestamp: datetime
    reasons: List[str]


class SignalGenerator:
    """Combines ML prediction with indicator confluence for signal generation."""

    def __init__(self, config: SignalConfig) -> None:
        self._config = config
        self._history: List[Signal] = []

    def generate(
        self,
        current_price: float,
        predicted_price: float,
        model_confidence: float,
        individual_preds: Dict[str, float],
        indicator_row: pd.Series,
    ) -> Signal:
        """Generate a signal combining ML prediction with indicator confluence."""
        change_pct = ((predicted_price - current_price) / current_price) * 100.0

        # Model directional agreement
        directions = [1 if p > current_price else -1 for p in individual_preds.values()]
        model_agreement = abs(sum(directions)) / len(directions) if directions else 0.0

        # Indicator confluence
        confluence_count, confluence_reasons = self._check_confluence(
            change_pct, current_price, indicator_row,
        )

        # Combined confidence score
        max_confluence = 5
        confluence_score = confluence_count / max_confluence

        combined_confidence = (
            0.5 * model_confidence
            + 0.3 * (model_agreement * 100.0)
            + 0.2 * (confluence_score * 100.0)
        )
        combined_confidence = max(0.0, min(100.0, combined_confidence))

        # Risk/reward (ATR-based)
        atr = indicator_row.get("ATR", abs(change_pct / 100.0 * current_price))
        if pd.isna(atr) or atr <= 0:
            atr = abs(change_pct / 100.0 * current_price) or 1.0
        reward = abs(predicted_price - current_price)
        risk_reward = reward / atr if atr > 0 else 0.0

        direction = self._classify(change_pct, combined_confidence)

        signal = Signal(
            direction=direction,
            confidence=round(combined_confidence, 1),
            predicted_price=predicted_price,
            current_price=current_price,
            change_pct=round(change_pct, 4),
            model_agreement=round(model_agreement, 2),
            indicator_confluence=confluence_count,
            risk_reward=round(risk_reward, 2),
            timestamp=datetime.now(),
            reasons=confluence_reasons,
        )

        self._history.append(signal)
        logger.info(
            "Signal: %s (confidence=%.1f%%, change=%.2f%%)",
            signal.direction, signal.confidence, signal.change_pct,
        )
        return signal

    def update_thresholds(
        self,
        buy: float,
        sell: float,
        strong_buy: float,
        strong_sell: float,
    ) -> None:
        """Update signal thresholds (e.g. from backtest optimization)."""
        # Create a new config since it's frozen - we store mutable overrides
        self._buy = buy
        self._sell = sell
        self._strong_buy = strong_buy
        self._strong_sell = strong_sell
        self._custom_thresholds = True
        logger.info(
            "Thresholds updated: SB=%.1f, B=%.1f, S=%.1f, SS=%.1f",
            strong_buy, buy, sell, strong_sell,
        )

    @property
    def history(self) -> List[Signal]:
        return list(self._history)

    def generate_regime_adaptive(
        self,
        current_price: float,
        predicted_price: float,
        model_confidence: float,
        individual_preds: Dict[str, float],
        indicator_row: pd.Series,
    ) -> Signal:
        """Generate signal with regime-adaptive thresholds.

        In trending markets: lower thresholds (easier to trigger, ride the trend)
        In choppy markets: higher thresholds (harder to trigger, avoid whipsaws)
        """
        # Detect regime from indicators
        efficiency = indicator_row.get("efficiency_ratio", 0.5)
        adx = indicator_row.get("ADX", 25)
        hurst = indicator_row.get("hurst_exponent", 0.5)

        if pd.isna(efficiency):
            efficiency = 0.5
        if pd.isna(adx):
            adx = 25
        if pd.isna(hurst):
            hurst = 0.5

        # Regime multiplier: trending = smaller thresholds, choppy = larger
        # Range: 0.5 (very trending) to 2.0 (very choppy)
        trending_score = (float(efficiency) + float(hurst) + float(adx) / 50) / 3
        regime_mult = max(0.5, min(2.0, 2.0 - trending_score * 1.5))

        # Temporarily adjust thresholds
        orig = self._config
        self._buy = orig.buy_threshold * regime_mult
        self._sell = orig.sell_threshold * regime_mult
        self._strong_buy = orig.strong_buy_threshold * regime_mult
        self._strong_sell = orig.strong_sell_threshold * regime_mult
        self._custom_thresholds = True

        # Generate signal with adjusted thresholds
        signal = self.generate(
            current_price, predicted_price, model_confidence,
            individual_preds, indicator_row,
        )

        # Restore original
        self._custom_thresholds = False

        return signal

    def _classify(self, change_pct: float, confidence: float) -> str:
        """Classify signal direction with minimum confidence gate."""
        if confidence < self._config.min_confidence:
            return "HOLD"

        # Use custom thresholds if set by backtest
        if getattr(self, "_custom_thresholds", False):
            sb, b, s, ss = self._strong_buy, self._buy, self._sell, self._strong_sell
        else:
            sc = self._config
            sb, b, s, ss = (
                sc.strong_buy_threshold, sc.buy_threshold,
                sc.sell_threshold, sc.strong_sell_threshold,
            )

        if change_pct > sb:
            return "STRONG BUY"
        if change_pct > b:
            return "BUY"
        if change_pct < ss:
            return "STRONG SELL"
        if change_pct < s:
            return "SELL"
        return "HOLD"

    def _check_confluence(
        self, change_pct: float, current_price: float, row: pd.Series,
    ) -> tuple[int, list[str]]:
        """Count how many indicators confirm the predicted direction."""
        count = 0
        reasons: list[str] = []

        rsi = row.get("RSI", 50.0)
        macd = row.get("MACD", 0.0)
        macd_sig = row.get("MACD_signal", 0.0)
        ma20 = row.get("ma20", current_price)
        bb_upper = row.get("BB_upper", current_price * 1.05)
        bb_lower = row.get("BB_lower", current_price * 0.95)
        stoch_k = row.get("stoch_k", 50.0)

        # Safe NaN handling
        for val_name in ("rsi", "macd", "macd_sig", "ma20", "stoch_k"):
            val = locals()[val_name]
            if pd.isna(val):
                locals()[val_name] = {"rsi": 50, "macd": 0, "macd_sig": 0,
                                       "ma20": current_price, "stoch_k": 50}[val_name]

        if change_pct > 0:  # Bullish
            if rsi < self._config.rsi_overbought:
                count += 1
                reasons.append(f"RSI not overbought ({rsi:.1f})")
            if macd > macd_sig:
                count += 1
                reasons.append("MACD above signal line")
            if current_price > ma20:
                count += 1
                reasons.append("Price above MA20")
            if current_price < bb_upper:
                count += 1
                reasons.append("Price below upper BB")
            if stoch_k < 80:
                count += 1
                reasons.append(f"Stochastic not overbought ({stoch_k:.1f})")
        else:  # Bearish
            if rsi > self._config.rsi_oversold:
                count += 1
                reasons.append(f"RSI not oversold ({rsi:.1f})")
            if macd < macd_sig:
                count += 1
                reasons.append("MACD below signal line")
            if current_price < ma20:
                count += 1
                reasons.append("Price below MA20")
            if current_price > bb_lower:
                count += 1
                reasons.append("Price above lower BB")
            if stoch_k > 20:
                count += 1
                reasons.append(f"Stochastic not oversold ({stoch_k:.1f})")

        return count, reasons
