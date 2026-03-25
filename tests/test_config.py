"""Tests for configuration."""

from btcdump.config import AppConfig, FeatureConfig, ModelConfig, SignalConfig


class TestConfig:
    def test_default_config_creates(self):
        cfg = AppConfig()
        assert cfg is not None

    def test_feature_count_over_100(self):
        cfg = AppConfig()
        assert len(cfg.features.feature_columns) >= 100

    def test_no_duplicate_features(self):
        cfg = AppConfig()
        features = cfg.features.feature_columns
        assert len(features) == len(set(features)), "Duplicate features found"

    def test_window_size_positive(self):
        cfg = AppConfig()
        assert cfg.features.window_size > 0

    def test_signal_thresholds_ordered(self):
        cfg = AppConfig()
        assert cfg.signal.strong_sell_threshold < cfg.signal.sell_threshold
        assert cfg.signal.sell_threshold < cfg.signal.buy_threshold
        assert cfg.signal.buy_threshold < cfg.signal.strong_buy_threshold

    def test_model_params_present(self):
        cfg = AppConfig()
        assert "n_estimators" in cfg.model.xgb_params
        assert "n_estimators" in cfg.model.rf_params
        assert "n_estimators" in cfg.model.gb_params

    def test_walk_forward_folds_positive(self):
        cfg = AppConfig()
        assert cfg.model.walk_forward_folds >= 2

    def test_rsi_thresholds_valid(self):
        cfg = AppConfig()
        assert 0 < cfg.signal.rsi_oversold < cfg.signal.rsi_overbought < 100

    def test_default_watchlist_not_empty(self):
        cfg = AppConfig()
        assert len(cfg.data.default_watchlist) > 0
        assert all(s.endswith("USDT") for s in cfg.data.default_watchlist)
