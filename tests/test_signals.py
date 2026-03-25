"""Tests for signal generation."""

import numpy as np
import pandas as pd
import pytest

from btcdump.config import AppConfig, SignalConfig
from btcdump.signals import SignalGenerator


@pytest.fixture
def signal_gen():
    return SignalGenerator(SignalConfig())


@pytest.fixture
def indicator_row():
    return pd.Series({
        "RSI": 55, "MACD": 0.5, "MACD_signal": 0.3,
        "ma20": 50000, "BB_upper": 52000, "BB_lower": 48000,
        "stoch_k": 60, "ATR": 500, "ADX": 30,
        "efficiency_ratio": 0.6, "hurst_exponent": 0.55,
    })


class TestSignalGenerator:
    def test_buy_signal(self, signal_gen, indicator_row):
        sig = signal_gen.generate(
            current_price=50000, predicted_price=50500,
            model_confidence=70, individual_preds={"xgb": 50600, "rf": 50400, "gb": 50500},
            indicator_row=indicator_row,
        )
        assert sig.direction in ("BUY", "STRONG BUY")
        assert sig.confidence > 0

    def test_sell_signal(self, signal_gen, indicator_row):
        sig = signal_gen.generate(
            current_price=50000, predicted_price=49000,
            model_confidence=70, individual_preds={"xgb": 48900, "rf": 49100, "gb": 49000},
            indicator_row=indicator_row,
        )
        assert sig.direction in ("SELL", "STRONG SELL")

    def test_hold_signal_low_confidence(self, signal_gen, indicator_row):
        sig = signal_gen.generate(
            current_price=50000, predicted_price=50010,
            model_confidence=10, individual_preds={"xgb": 50020, "rf": 49990, "gb": 50010},
            indicator_row=indicator_row,
        )
        assert sig.direction == "HOLD"

    def test_model_agreement(self, signal_gen, indicator_row):
        # All models agree on direction
        sig = signal_gen.generate(
            current_price=50000, predicted_price=51000,
            model_confidence=80, individual_preds={"xgb": 51000, "rf": 51200, "gb": 50800},
            indicator_row=indicator_row,
        )
        assert sig.model_agreement == 1.0

    def test_model_disagreement(self, signal_gen, indicator_row):
        sig = signal_gen.generate(
            current_price=50000, predicted_price=50100,
            model_confidence=60, individual_preds={"xgb": 50200, "rf": 49800, "gb": 50100},
            indicator_row=indicator_row,
        )
        assert sig.model_agreement < 1.0

    def test_indicator_confluence(self, signal_gen, indicator_row):
        sig = signal_gen.generate(
            current_price=50000, predicted_price=51000,
            model_confidence=80, individual_preds={"xgb": 51000, "rf": 51200, "gb": 50800},
            indicator_row=indicator_row,
        )
        assert sig.indicator_confluence >= 0
        assert sig.indicator_confluence <= 5

    def test_risk_reward_positive(self, signal_gen, indicator_row):
        sig = signal_gen.generate(
            current_price=50000, predicted_price=51000,
            model_confidence=80, individual_preds={"xgb": 51000, "rf": 51200, "gb": 50800},
            indicator_row=indicator_row,
        )
        assert sig.risk_reward >= 0

    def test_confidence_range(self, signal_gen, indicator_row):
        sig = signal_gen.generate(
            current_price=50000, predicted_price=51000,
            model_confidence=80, individual_preds={"xgb": 51000, "rf": 51200, "gb": 50800},
            indicator_row=indicator_row,
        )
        assert 0 <= sig.confidence <= 100

    def test_reasons_are_strings(self, signal_gen, indicator_row):
        sig = signal_gen.generate(
            current_price=50000, predicted_price=51000,
            model_confidence=80, individual_preds={"xgb": 51000, "rf": 51200, "gb": 50800},
            indicator_row=indicator_row,
        )
        assert all(isinstance(r, str) for r in sig.reasons)

    def test_regime_adaptive_generates(self, signal_gen, indicator_row):
        sig = signal_gen.generate_regime_adaptive(
            current_price=50000, predicted_price=51000,
            model_confidence=80, individual_preds={"xgb": 51000, "rf": 51200, "gb": 50800},
            indicator_row=indicator_row,
        )
        assert sig.direction in ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")
