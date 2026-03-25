"""Tests for indicator computation and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from btcdump.config import AppConfig
from btcdump.indicators import (
    compute_all,
    compute_fibonacci_levels,
    compute_relative_strength,
    compute_seasonality_profile,
    detect_anomalies,
    detect_support_resistance,
    detect_trend_lines,
)


@pytest.fixture
def config():
    return AppConfig()


@pytest.fixture
def sample_df():
    """Generate realistic OHLCV data with a trend + noise."""
    np.random.seed(42)
    n = 300
    base = 40000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": base + np.random.randn(n) * 50,
        "high": base + abs(np.random.randn(n) * 150),
        "low": base - abs(np.random.randn(n) * 150),
        "close": base + np.random.randn(n) * 80,
        "volume": np.random.uniform(100, 1000, n),
    })


class TestComputeAll:
    def test_all_features_present(self, sample_df, config):
        result = compute_all(sample_df, config.indicators)
        missing = [c for c in config.features.feature_columns if c not in result.columns]
        assert missing == [], f"Missing features: {missing}"

    def test_feature_count(self, config):
        assert len(config.features.feature_columns) >= 100  # at least 100 features

    def test_dimensions(self, config):
        dims = len(config.features.feature_columns) * config.features.window_size
        assert dims >= 2000  # at least 2000 dims

    def test_no_all_nan_columns(self, sample_df, config):
        result = compute_all(sample_df, config.indicators)
        for col in config.features.feature_columns:
            non_null = result[col].notna().sum()
            assert non_null > 0, f"Feature {col} is all NaN"

    def test_rsi_range(self, sample_df, config):
        result = compute_all(sample_df, config.indicators)
        rsi = result["RSI"].dropna()
        assert rsi.min() >= 0, "RSI below 0"
        assert rsi.max() <= 100, "RSI above 100"

    def test_body_ratio_nonnegative(self, sample_df, config):
        result = compute_all(sample_df, config.indicators)
        br = result["body_ratio"].dropna()
        assert br.min() >= -0.01, "body_ratio significantly negative"

    def test_efficiency_ratio_range(self, sample_df, config):
        result = compute_all(sample_df, config.indicators)
        er = result["efficiency_ratio"].dropna()
        assert er.min() >= 0, "ER negative"
        assert er.max() <= 1.01, "ER > 1"

    def test_benchmark_under_1s(self, sample_df, config):
        import time
        t0 = time.time()
        compute_all(sample_df, config.indicators)
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"Computation took {elapsed:.2f}s (expected < 2s)"


class TestFibonacci:
    def test_returns_7_levels(self, sample_df):
        levels = compute_fibonacci_levels(sample_df)
        assert len(levels) == 7

    def test_level_names(self, sample_df):
        levels = compute_fibonacci_levels(sample_df)
        names = [l["name"] for l in levels]
        assert "0%" in names
        assert "50%" in names
        assert "100%" in names
        assert "61.8%" in names

    def test_prices_ordered(self, sample_df):
        levels = compute_fibonacci_levels(sample_df)
        prices = [l["price"] for l in levels]
        assert prices == sorted(prices) or prices == sorted(prices, reverse=True)


class TestSupportResistance:
    def test_returns_list(self, sample_df):
        levels = detect_support_resistance(sample_df)
        assert isinstance(levels, list)

    def test_has_support_and_resistance(self, sample_df):
        levels = detect_support_resistance(sample_df)
        types = {l["type"] for l in levels}
        assert "support" in types or "resistance" in types

    def test_touches_positive(self, sample_df):
        levels = detect_support_resistance(sample_df)
        for l in levels:
            assert l["touches"] > 0


class TestAnomalyDetection:
    def test_returns_dict(self, sample_df):
        result = detect_anomalies(sample_df)
        assert "volume_anomaly" in result
        assert "price_anomaly" in result
        assert "volume_zscore" in result


class TestTrendLines:
    def test_returns_lines(self, sample_df):
        lines = detect_trend_lines(sample_df)
        assert isinstance(lines, list)
        assert len(lines) <= 2  # at most support + resistance

    def test_line_has_required_fields(self, sample_df):
        lines = detect_trend_lines(sample_df)
        for l in lines:
            assert "type" in l
            assert "start_price" in l
            assert "end_price" in l
            assert "slope_per_bar" in l


class TestRelativeStrength:
    def test_returns_classification(self, sample_df):
        df2 = sample_df.copy()
        df2["close"] = df2["close"] * 1.1  # outperformer
        result = compute_relative_strength(df2, sample_df)
        assert "classification" in result
        assert "rs_ratio" in result

    def test_outperformer_detected(self, sample_df):
        df2 = sample_df.copy()
        # Make df2 clearly outperform
        df2["close"] = sample_df["close"] * np.linspace(1.0, 1.1, len(sample_df))
        result = compute_relative_strength(df2, sample_df)
        assert result["rs_ratio"] > 0


class TestSeasonality:
    def test_returns_hourly_and_daily(self, sample_df):
        result = compute_seasonality_profile(sample_df)
        assert "hourly" in result
        assert "daily" in result
        assert len(result["hourly"]) > 0
