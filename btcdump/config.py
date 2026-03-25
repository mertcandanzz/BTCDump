"""Centralized configuration - all magic numbers live here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class DataConfig:
    base_url: str = "https://api.binance.com/api/v3/klines"
    binance_api_base: str = "https://api.binance.com"
    default_symbol: str = "BTCUSDT"
    default_interval: str = "1h"
    default_watchlist: Tuple[str, ...] = (
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    )
    candle_limit: int = 1000
    mini_chart_candles: int = 50
    request_timeout: int = 10
    max_retries: int = 3
    retry_backoff: float = 1.5
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    cache_ttl_seconds: int = 300
    exchange_info_cache_ttl: int = 3600
    ticker_cache_ttl: int = 60
    signal_workers: int = 4


@dataclass(frozen=True)
class IndicatorConfig:
    ma_periods: Tuple[int, ...] = (5, 20, 50)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    adx_period: int = 14


@dataclass(frozen=True)
class FeatureConfig:
    window_size: int = 20
    feature_columns: Tuple[str, ...] = (
        # Core OHLCV
        "close", "volume",
        # Classic indicators
        "RSI", "MACD", "volume_ratio",
        "ma5", "ma20", "ma50", "BB_upper", "BB_lower",
        "ATR", "stoch_k", "stoch_d", "ADX", "OBV_norm",
        "ROC", "williams_r", "CCI", "MFI",
        # Returns & momentum
        "returns_1", "returns_5", "price_momentum",
        # Volatility
        "volatility_10", "high_low_range", "parkinson_vol", "atr_ratio",
        # Candlestick body analysis
        "body_ratio", "buying_pressure", "upper_shadow", "lower_shadow",
        # Bollinger Band features
        "bb_pct_b", "bb_width",
        # Volume analysis
        "vol_delta", "ad_norm",
        # Trend strength
        "hh_streak", "ll_streak", "dist_from_high", "dist_from_low",
        # Momentum quality
        "rsi_momentum", "macd_hist_slope",
        # Mean reversion & position
        "close_ma_ratio", "price_zscore", "vwap_dist",
        # v5.0 Pro Features: Regime Detection
        "efficiency_ratio", "choppiness", "adx_slope",
        # v5.0 Pro: Advanced Momentum
        "tsi", "rsi_divergence", "returns_10", "returns_20", "momentum_quality",
        # v5.0 Pro: Volatility Regime
        "garch_proxy", "vol_of_vol", "yang_zhang_vol",
        # v5.0 Pro: Liquidity
        "volume_trend", "amihud_illiq",
        # v5.0 Pro: Time Cyclical
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        # v5.0 Pro: Distribution Shape
        "skewness_20", "kurtosis_20",
        # v5.0 Pro: Squeeze Detection
        "keltner_position", "squeeze_ratio",
        # v5.0 Pro: Pattern Quantification
        "engulfing_score", "consecutive_dir",
        # v5.0 Pro: Order Flow Proxy
        "ofi_14", "pv_divergence",
        # v5.0 Pro: Ichimoku Cloud
        "ichimoku_tk", "ichimoku_cloud_pos", "ichimoku_cloud_width",
        "ichimoku_chikou", "ichimoku_kijun_dist",
        # v5.0 Pro: Volume Profile
        "vp_poc_dist", "vp_va_position",
        # v5.0 Pro: Pivot Points
        "pivot_dist", "pivot_r1_dist", "pivot_s1_dist", "pivot_position",
        # v5.0 Pro: Candlestick Patterns
        "pattern_doji", "pattern_hammer", "pattern_shooting_star",
        "pattern_three_soldiers", "pattern_three_crows",
        "pattern_morning_star", "pattern_evening_star",
        # v5.0 Pro: Microstructure
        "trade_intensity", "pin_bar_score", "gap_pct",
        "intrabar_vol_ratio", "close_position_avg",
        # v5.0 Pro: Statistical
        "hurst_exponent", "autocorr_1", "autocorr_5",
        "di_ratio", "di_spread", "variance_ratio",
        # v5.0 Pro: Whale/Smart Money
        "whale_score", "smart_money_div",
        # v5.0 Pro: Information Theory
        "price_entropy",
        # v5.0 Pro: Adaptive Moving Averages
        "kama_dist", "dema_dist", "tema_dist", "kama_slope",
        # v5.0 Pro: Seasonality
        "seasonal_hour_bias", "seasonal_dow_bias",
        # v5.0 Pro: Cycle Detection
        "cycle_phase", "cycle_strength", "dominant_period",
        # v5.0 Pro: Information Theory
        "transfer_entropy", "mutual_info_pv",
        # v5.0 Pro: Complexity
        "approx_entropy", "sample_entropy_proxy",
    )


@dataclass(frozen=True)
class ModelConfig:
    models_dir: Path = field(default_factory=lambda: Path("data/models"))
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200, "learning_rate": 0.05,
        "max_depth": 6, "subsample": 0.8, "random_state": 42,
    })
    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200, "max_depth": 10,
        "min_samples_split": 5, "random_state": 42,
    })
    gb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200, "learning_rate": 0.05,
        "max_depth": 5, "random_state": 42,
    })
    walk_forward_folds: int = 5
    min_train_size: int = 200
    test_size: int = 50
    retrain_interval_candles: int = 10


@dataclass(frozen=True)
class SignalConfig:
    strong_buy_threshold: float = 1.5
    buy_threshold: float = 0.5
    sell_threshold: float = -0.5
    strong_sell_threshold: float = -1.5
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    min_confidence: float = 30.0


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: Path("data/btcdump.log"))
