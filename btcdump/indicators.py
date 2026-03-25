"""Technical indicator calculations - pure, stateless functions.

Every function takes a DataFrame, returns a new DataFrame with added columns.
NaN rows are NOT dropped here; the caller decides how to handle warmup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcdump.config import IndicatorConfig


def compute_all(df: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
    """Compute all configured indicators on an OHLCV DataFrame."""
    df = df.copy()
    df = _moving_averages(df, config.ma_periods)
    df = _rsi(df, config.rsi_period)
    df = _macd(df, config.macd_fast, config.macd_slow, config.macd_signal)
    df = _bollinger_bands(df, config.bb_period, config.bb_std)
    df = _atr(df, config.atr_period)
    df = _stochastic(df, config.stoch_k_period, config.stoch_d_period)
    df = _adx(df, config.adx_period)
    df = _obv(df)
    df = _vwap(df)
    df = _volume_features(df)
    df = _roc(df)
    df = _williams_r(df)
    df = _cci(df)
    df = _mfi(df)
    df = _derived_features(df)
    df = _regime_features(df)
    df = _advanced_momentum(df)
    df = _volatility_regime(df)
    df = _liquidity_features(df)
    df = _time_features(df)
    df = _distribution_features(df)
    df = _squeeze_features(df)
    df = _pattern_features(df)
    df = _ichimoku(df)
    df = _volume_profile(df)
    df = _pivot_points(df)
    df = _candlestick_patterns(df)
    df = _microstructure_features(df)
    df = df.copy()  # defragment mid-pipeline to avoid PerformanceWarning
    df = _statistical_features(df)
    df = _whale_detection(df)
    df = _entropy_feature(df)
    df = _adaptive_ma_features(df)
    df = _seasonality_features(df)
    df = _cycle_features(df)
    df = _information_theory_features(df)
    df = _complexity_features(df)
    df = _dfa_feature(df)
    df = _interaction_features(df)
    df = _jump_detection(df)
    df = _kalman_features(df)
    df = _ou_features(df)
    df = _final_features(df)
    return df.copy()  # final defragment


# ---------------------------------------------------------------------------
# Individual indicator implementations
# ---------------------------------------------------------------------------

def _moving_averages(df: pd.DataFrame, periods: tuple[int, ...]) -> pd.DataFrame:
    for p in periods:
        df[f"ma{p}"] = df["close"].rolling(p).mean()
    return df


def _rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Wilder's RSI using EWM with com=period-1."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def _macd(
    df: pd.DataFrame, fast: int, slow: int, signal: int,
) -> pd.DataFrame:
    exp_fast = df["close"].ewm(span=fast, adjust=False).mean()
    exp_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = exp_fast - exp_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def _bollinger_bands(df: pd.DataFrame, period: int, std_mult: float) -> pd.DataFrame:
    df["BB_middle"] = df["close"].rolling(period).mean()
    bb_std = df["close"].rolling(period).std()
    df["BB_upper"] = df["BB_middle"] + (bb_std * std_mult)
    df["BB_lower"] = df["BB_middle"] - (bb_std * std_mult)
    return df


def _atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(period).mean()
    return df


def _stochastic(df: pd.DataFrame, k_period: int, d_period: int) -> pd.DataFrame:
    """Stochastic Oscillator %K and %D."""
    lowest = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    denom = highest - lowest
    df["stoch_k"] = 100.0 * (df["close"] - lowest) / denom.replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def _adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Average Directional Index."""
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    atr = df["ATR"] if "ATR" in df.columns else _atr(df.copy(), period)["ATR"]

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.rolling(period).mean()
    return df


def _obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume, normalized to z-score."""
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).cumsum()
    roll_mean = obv.rolling(50, min_periods=1).mean()
    roll_std = obv.rolling(50, min_periods=1).std().replace(0, 1.0)
    df["OBV_norm"] = (obv - roll_mean) / roll_std
    return df


def _vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling VWAP proxy (crypto has no session boundaries)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol_sum = df["volume"].rolling(20, min_periods=1).sum()
    df["VWAP"] = (
        (typical_price * df["volume"]).rolling(20, min_periods=1).sum()
        / vol_sum.replace(0, np.nan)
    )
    return df


def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
    return df


def _roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Rate of Change - momentum indicator."""
    df["ROC"] = df["close"].pct_change(period) * 100
    return df


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R - overbought/oversold oscillator."""
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    denom = (highest - lowest).replace(0, np.nan)
    df["williams_r"] = -100 * (highest - df["close"]) / denom
    return df


def _cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    return df


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Money Flow Index - volume-weighted RSI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    df["MFI"] = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))
    return df


def _derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive daytrader features: price action, momentum, volatility, volume."""
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # ── Returns & Momentum ──
    df["returns_1"] = c.pct_change(1)
    df["returns_5"] = c.pct_change(5)
    df["price_momentum"] = c - c.shift(10)

    # ── Volatility ──
    df["volatility_10"] = c.pct_change().rolling(10).std()
    df["high_low_range"] = (h - l) / c.replace(0, np.nan)

    # MA ratio (price position relative to MA20)
    ma20 = df.get("ma20")
    if ma20 is not None:
        df["close_ma_ratio"] = c / ma20.replace(0, np.nan) - 1
    else:
        df["close_ma_ratio"] = 0.0

    # ── Candlestick Body Analysis ──
    body = (c - o).abs()
    full_range = (h - l).replace(0, np.nan)

    # Body-to-range ratio: 1.0 = full body (strong conviction), 0.0 = doji (indecision)
    df["body_ratio"] = body / full_range

    # Buying pressure: where close sits in the high-low range (0=at low, 1=at high)
    df["buying_pressure"] = (c - l) / full_range

    # Upper shadow ratio: rejection from highs (shooting star pattern)
    df["upper_shadow"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / full_range

    # Lower shadow ratio: rejection from lows (hammer pattern)
    df["lower_shadow"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / full_range

    # ── Bollinger Band Features ──
    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
        # %B: where price is within the bands (0=lower, 1=upper, >1=above upper)
        df["bb_pct_b"] = (c - df["BB_lower"]) / bb_range
        # BB Width: squeeze detection (narrow = breakout incoming)
        df["bb_width"] = bb_range / df["BB_middle"].replace(0, np.nan)
    else:
        df["bb_pct_b"] = 0.5
        df["bb_width"] = 0.0

    # ── Volatility: Parkinson (high-low based, better estimator) ──
    df["parkinson_vol"] = np.sqrt(
        (np.log(h / l.replace(0, np.nan)) ** 2).rolling(20).mean() / (4 * np.log(2))
    )

    # ATR ratio: expanding vs contracting volatility
    if "ATR" in df.columns:
        atr_slow = df["ATR"].rolling(50, min_periods=1).mean()
        df["atr_ratio"] = df["ATR"] / atr_slow.replace(0, np.nan)
    else:
        df["atr_ratio"] = 1.0

    # ── Volume Analysis ──
    # Volume delta proxy: positive volume = bullish candle volume
    direction = np.sign(c - o)
    df["vol_delta"] = (direction * v).rolling(14).sum() / v.rolling(14).sum().replace(0, np.nan)

    # Accumulation/Distribution proxy
    mf_multiplier = ((c - l) - (h - c)) / full_range
    df["ad_line"] = (mf_multiplier.fillna(0) * v).cumsum()
    ad = df["ad_line"]
    ad_mean = ad.rolling(20, min_periods=1).mean()
    ad_std = ad.rolling(20, min_periods=1).std().replace(0, 1.0)
    df["ad_norm"] = (ad - ad_mean) / ad_std

    # ── Trend Strength ──
    # Higher highs / lower lows streak
    df["hh_streak"] = (h > h.shift(1)).astype(int).groupby(
        (~(h > h.shift(1))).cumsum()
    ).cumsum()
    df["ll_streak"] = (l < l.shift(1)).astype(int).groupby(
        (~(l < l.shift(1))).cumsum()
    ).cumsum()

    # Distance from 20-period high/low (support/resistance proximity)
    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    range_20 = (high_20 - low_20).replace(0, np.nan)
    df["dist_from_high"] = (high_20 - c) / range_20  # 0 = at high, 1 = at low
    df["dist_from_low"] = (c - low_20) / range_20   # 0 = at low, 1 = at high

    # ── Momentum Quality ──
    # RSI momentum (acceleration of RSI)
    if "RSI" in df.columns:
        df["rsi_momentum"] = df["RSI"].diff(3)
    else:
        df["rsi_momentum"] = 0.0

    # MACD histogram slope (momentum of momentum)
    if "MACD_hist" in df.columns:
        df["macd_hist_slope"] = df["MACD_hist"].diff(3)
    else:
        df["macd_hist_slope"] = 0.0

    # ── Mean Reversion ──
    # Z-score of price (how many std devs from 20-period mean)
    mean_20 = c.rolling(20).mean()
    std_20 = c.rolling(20).std().replace(0, np.nan)
    df["price_zscore"] = (c - mean_20) / std_20

    # ── VWAP Distance ──
    if "VWAP" in df.columns:
        atr_val = df.get("ATR", pd.Series(1.0, index=df.index))
        df["vwap_dist"] = (c - df["VWAP"]) / atr_val.replace(0, np.nan)
    else:
        df["vwap_dist"] = 0.0

    return df


def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Regime detection: trending vs mean-reverting."""
    c = df["close"]

    # Kaufman Efficiency Ratio: |net move| / sum(|bar moves|)
    n = 10
    net_change = (c - c.shift(n)).abs()
    volatility_sum = c.diff().abs().rolling(n).sum()
    df["efficiency_ratio"] = net_change / volatility_sum.replace(0, np.nan)

    # Choppiness Index: 100 * log10(sum(ATR,14) / range) / log10(14)
    if "ATR" in df.columns:
        atr_sum = df["ATR"].rolling(14).sum()
        hh = df["high"].rolling(14).max()
        ll = df["low"].rolling(14).min()
        denom = (hh - ll).replace(0, np.nan)
        df["choppiness"] = 100 * np.log10(atr_sum / denom) / np.log10(14)
    else:
        df["choppiness"] = 50.0

    # ADX slope: trend strengthening or weakening
    if "ADX" in df.columns:
        df["adx_slope"] = df["ADX"].diff(5)
    else:
        df["adx_slope"] = 0.0

    return df


def _advanced_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced momentum: TSI, divergence, multi-scale returns."""
    c = df["close"]

    # True Strength Index (double-smoothed momentum)
    mom = c.diff(1)
    smooth1 = mom.ewm(span=25, adjust=False).mean()
    smooth2 = smooth1.ewm(span=13, adjust=False).mean()
    abs_smooth1 = mom.abs().ewm(span=25, adjust=False).mean()
    abs_smooth2 = abs_smooth1.ewm(span=13, adjust=False).mean()
    df["tsi"] = 100 * smooth2 / abs_smooth2.replace(0, np.nan)

    # RSI divergence (quantitative): price momentum vs RSI momentum
    if "RSI" in df.columns:
        df["rsi_divergence"] = c.pct_change(14) - df["RSI"].pct_change(14)
    else:
        df["rsi_divergence"] = 0.0

    # Medium-term returns
    df["returns_10"] = c.pct_change(10)
    df["returns_20"] = c.pct_change(20)

    # Momentum quality: return / volatility (directional Sharpe)
    vol_20 = c.pct_change().rolling(20).std().replace(0, np.nan)
    df["momentum_quality"] = c.pct_change(20) / vol_20

    return df


def _volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility regime: GARCH proxy, Yang-Zhang, vol-of-vol."""
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    ret = c.pct_change()

    # GARCH proxy: short-term vol / long-term vol
    realized_5 = ret.rolling(5).std()
    expected_20 = ret.rolling(20).std().replace(0, np.nan)
    df["garch_proxy"] = realized_5 / expected_20

    # Vol of vol (convexity)
    vol_10 = ret.rolling(10).std()
    df["vol_of_vol"] = vol_10.rolling(10).std()

    # Yang-Zhang volatility (state-of-the-art OHLCV estimator)
    overnight = np.log(o / c.shift(1).replace(0, np.nan))
    intraday_oc = np.log(c / o.replace(0, np.nan))
    intraday_hl = np.log(h / l.replace(0, np.nan))

    var_o = overnight.rolling(20).var()
    var_oc = intraday_oc.rolling(20).var()
    var_hl = intraday_hl.rolling(20).var()
    k = 0.34 / (1.34 + (20 + 1) / (20 - 1))
    yz_var = var_o + k * var_oc + (1 - k) * var_hl
    df["yang_zhang_vol"] = np.sqrt(yz_var.clip(lower=0))

    return df


def _liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Liquidity: volume trend, Amihud illiquidity."""
    v = df["volume"]
    c = df["close"]
    ret = c.pct_change().abs()

    # Volume trend (slope of volume / mean volume)
    v_mean = v.rolling(20, min_periods=1).mean()
    v_diff = v.diff()
    df["volume_trend"] = v_diff.rolling(20, min_periods=1).mean() / v_mean.replace(0, np.nan)

    # Amihud illiquidity: |return| / dollar volume
    dollar_vol = (v * c).replace(0, np.nan)
    amihud = (ret / dollar_vol)
    # Log transform and rolling mean for stability
    df["amihud_illiq"] = np.log1p(amihud.rolling(20, min_periods=1).mean() * 1e6)

    return df


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time features: hour, day of week."""
    if "time" not in df.columns:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 0.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 0.0
        return df

    try:
        ts = pd.to_datetime(df["time"])
        hour = ts.dt.hour + ts.dt.minute / 60.0
        dow = ts.dt.dayofweek

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    except Exception:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 0.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 0.0

    return df


def _distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return distribution shape: skewness, kurtosis."""
    ret = df["close"].pct_change()
    df["skewness_20"] = ret.rolling(20).skew()
    df["kurtosis_20"] = ret.rolling(20).kurt()
    return df


def _squeeze_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keltner channels + Bollinger squeeze detection."""
    c = df["close"]

    # Keltner channels (EMA-based with ATR)
    keltner_mid = c.ewm(span=20, adjust=False).mean()
    if "ATR" in df.columns:
        keltner_upper = keltner_mid + 2.0 * df["ATR"]
        keltner_lower = keltner_mid - 2.0 * df["ATR"]
    else:
        keltner_upper = keltner_mid * 1.02
        keltner_lower = keltner_mid * 0.98

    keltner_range = (keltner_upper - keltner_lower).replace(0, np.nan)
    df["keltner_position"] = (c - keltner_lower) / keltner_range

    # BB squeeze: BB width / Keltner width
    if "bb_width" in df.columns:
        bb_range = df.get("BB_upper", keltner_upper) - df.get("BB_lower", keltner_lower)
        df["squeeze_ratio"] = bb_range / keltner_range
    else:
        df["squeeze_ratio"] = 1.0

    return df


def _pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-candle pattern quantification."""
    c, o = df["close"], df["open"]

    # Engulfing score
    prev_body = c.shift(1) - o.shift(1)
    curr_body = c - o
    abs_prev = prev_body.abs().replace(0, np.nan)
    bull_engulf = (curr_body > 0) & (prev_body < 0) & (curr_body.abs() > abs_prev)
    bear_engulf = (curr_body < 0) & (prev_body > 0) & (curr_body.abs() > abs_prev)
    df["engulfing_score"] = np.where(
        bull_engulf, curr_body / abs_prev,
        np.where(bear_engulf, curr_body / abs_prev, 0),
    )

    # Consecutive direction count (positive=bullish streak, negative=bearish)
    direction = np.sign(c - o)
    groups = (direction != direction.shift()).cumsum()
    streak = direction.groupby(groups).cumcount() + 1
    df["consecutive_dir"] = streak * direction

    # ── Order Flow Imbalance (OFI) proxy ──
    # Buying pressure intensity: how aggressively price moved to close
    h, l, v = df["high"], df["low"], df["volume"]
    full_range = (h - l).replace(0, np.nan)

    # Imbalance: buy volume fraction minus sell volume fraction
    buy_pct = (c - l) / full_range       # 0=closed at low, 1=closed at high
    sell_pct = (h - c) / full_range       # 0=closed at high, 1=closed at low
    raw_imbalance = (buy_pct - sell_pct) * v  # signed volume imbalance

    # Cumulative Order Flow Imbalance (rolling)
    df["ofi_14"] = raw_imbalance.rolling(14).sum() / v.rolling(14).sum().replace(0, np.nan)

    # Price-Volume Divergence: strong move on weak volume or weak move on strong volume
    ret_abs = c.pct_change().abs()
    vol_ratio = v / v.rolling(20, min_periods=1).mean().replace(0, np.nan)
    df["pv_divergence"] = ret_abs / vol_ratio.replace(0, np.nan)  # high = thin move, low = volume-confirmed

    return df


def _ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Ichimoku Cloud: Tenkan/Kijun cross, cloud position, Chikou lag.

    Standard crypto settings: 20/60/120/30 (adapted from 9/26/52/26 for 24/7 markets).
    We use traditional settings for broader compatibility.
    """
    h, l, c = df["high"], df["low"], df["close"]

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26
    senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)

    # Features for ML:
    # 1. TK cross signal: Tenkan vs Kijun position (normalized)
    tk_diff = tenkan - kijun
    df["ichimoku_tk"] = tk_diff / c.replace(0, np.nan) * 100  # as % of price

    # 2. Price vs Cloud: where price sits relative to the cloud
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    cloud_mid = (cloud_top + cloud_bottom) / 2
    cloud_width = (cloud_top - cloud_bottom).replace(0, np.nan)
    df["ichimoku_cloud_pos"] = (c - cloud_mid) / cloud_width  # >0 = above cloud

    # 3. Cloud thickness (normalized): thin cloud = weak S/R
    df["ichimoku_cloud_width"] = cloud_width / c.replace(0, np.nan) * 100

    # 4. Chikou span position: current close vs close 26 bars ago
    df["ichimoku_chikou"] = (c - c.shift(26)) / c.shift(26).replace(0, np.nan) * 100

    # 5. Kijun distance: price distance from Kijun (mean-reversion signal)
    df["ichimoku_kijun_dist"] = (c - kijun) / c.replace(0, np.nan) * 100

    return df


def _volume_profile(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Volume Profile features: POC distance, Value Area position.

    Fast implementation: uses numpy histogram with volume weights.
    Computes every 5 bars and forward-fills for speed.
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    n = len(df)

    poc_dist = np.zeros(n)
    va_position = np.full(n, 0.5)

    window = 100
    step = 5  # compute every 5 bars (10x faster)

    for i in range(window, n, step):
        chunk_c = c.iloc[i - window:i].values
        chunk_h = h.iloc[i - window:i].values
        chunk_l = l.iloc[i - window:i].values
        chunk_v = v.iloc[i - window:i].values

        # Use close prices with volume weights for fast profile
        price_min, price_max = float(chunk_l.min()), float(chunk_h.max())
        if price_max <= price_min:
            continue

        profile, edges = np.histogram(chunk_c, bins=bins, range=(price_min, price_max), weights=chunk_v)

        # POC
        poc_bin = int(np.argmax(profile))
        poc_price = (edges[poc_bin] + edges[poc_bin + 1]) / 2
        current = float(chunk_c[-1])
        poc_val = (current - poc_price) / current * 100 if current > 0 else 0

        # Value Area (70%)
        total_vol = profile.sum()
        va_val = 0.5
        if total_vol > 0:
            sorted_idx = np.argsort(profile)[::-1]
            cum = 0.0
            va_lo_bin, va_hi_bin = bins, 0
            for sb in sorted_idx:
                cum += profile[sb]
                va_lo_bin = min(va_lo_bin, sb)
                va_hi_bin = max(va_hi_bin, sb)
                if cum >= total_vol * 0.7:
                    break
            va_low = edges[va_lo_bin]
            va_high = edges[va_hi_bin + 1]
            va_range = va_high - va_low
            if va_range > 0:
                va_val = (current - va_low) / va_range

        # Fill step range
        end = min(i + step, n)
        poc_dist[i:end] = poc_val
        va_position[i:end] = va_val

    df["vp_poc_dist"] = poc_dist
    df["vp_va_position"] = va_position
    return df


def _pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Classic Pivot Points: price position relative to daily pivots.

    Pivot = (High_prev + Low_prev + Close_prev) / 3
    R1 = 2*Pivot - Low_prev,  S1 = 2*Pivot - High_prev
    R2 = Pivot + (High_prev - Low_prev),  S2 = Pivot - (High_prev - Low_prev)

    We compute on rolling 24-bar basis (proxy for daily on 1h data)
    and normalize as distance from price.
    """
    c, h, l = df["close"], df["high"], df["low"]
    window = 24  # ~1 day on 1h data

    prev_h = h.rolling(window).max().shift(1)
    prev_l = l.rolling(window).min().shift(1)
    prev_c = c.shift(1)

    pivot = (prev_h + prev_l + prev_c) / 3
    r1 = 2 * pivot - prev_l
    s1 = 2 * pivot - prev_h
    r2 = pivot + (prev_h - prev_l)
    s2 = pivot - (prev_h - prev_l)

    # Normalize as % distance from current price
    df["pivot_dist"] = (c - pivot) / c.replace(0, np.nan) * 100
    df["pivot_r1_dist"] = (r1 - c) / c.replace(0, np.nan) * 100
    df["pivot_s1_dist"] = (c - s1) / c.replace(0, np.nan) * 100

    # Position within S2-R2 range (0 = at S2, 1 = at R2)
    pivot_range = (r2 - s2).replace(0, np.nan)
    df["pivot_position"] = (c - s2) / pivot_range

    return df


def _candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Quantified candlestick pattern detection.

    Returns scores (not booleans) so ML can learn pattern strength.
    """
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    body = c - o  # positive = bullish
    body_abs = body.abs()
    full_range = (h - l).replace(0, np.nan)
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l

    # ── Doji: tiny body relative to range ──
    # Score: 1.0 = perfect doji (body=0), 0.0 = full body
    doji_score = 1.0 - (body_abs / full_range).clip(0, 1)
    df["pattern_doji"] = doji_score

    # ── Hammer / Hanging Man: long lower wick, small body at top ──
    # Score: lower_wick > 2x body AND upper_wick < body
    hammer_cond = (lower_wick > body_abs * 2) & (upper_wick < body_abs * 0.5)
    df["pattern_hammer"] = np.where(
        hammer_cond,
        lower_wick / full_range,  # strength: how long is the wick
        0.0,
    )

    # ── Shooting Star / Inverted Hammer: long upper wick ──
    star_cond = (upper_wick > body_abs * 2) & (lower_wick < body_abs * 0.5)
    df["pattern_shooting_star"] = np.where(
        star_cond,
        upper_wick / full_range,
        0.0,
    )

    # ── Three White Soldiers: 3 consecutive bullish with higher closes ──
    bull1 = body.shift(2) > 0
    bull2 = body.shift(1) > 0
    bull3 = body > 0
    higher1 = c.shift(1) > c.shift(2)
    higher2 = c > c.shift(1)
    soldiers = bull1 & bull2 & bull3 & higher1 & higher2
    # Strength: average body size of the 3 candles
    avg_body = (body_abs + body_abs.shift(1) + body_abs.shift(2)) / 3
    df["pattern_three_soldiers"] = np.where(
        soldiers, avg_body / full_range, 0.0,
    )

    # ── Three Black Crows: opposite of soldiers ──
    bear1 = body.shift(2) < 0
    bear2 = body.shift(1) < 0
    bear3 = body < 0
    lower1 = c.shift(1) < c.shift(2)
    lower2 = c < c.shift(1)
    crows = bear1 & bear2 & bear3 & lower1 & lower2
    df["pattern_three_crows"] = np.where(
        crows, avg_body / full_range, 0.0,
    )

    # ── Morning Star: bearish + doji/small + bullish (reversal) ──
    morning = (body.shift(2) < 0) & (body_abs.shift(1) < body_abs.shift(2) * 0.3) & (body > 0) & (c > (o.shift(2) + c.shift(2)) / 2)
    df["pattern_morning_star"] = np.where(morning, body_abs / full_range, 0.0)

    # ── Evening Star: bullish + doji/small + bearish (reversal) ──
    evening = (body.shift(2) > 0) & (body_abs.shift(1) < body_abs.shift(2) * 0.3) & (body < 0) & (c < (o.shift(2) + c.shift(2)) / 2)
    df["pattern_evening_star"] = np.where(evening, body_abs / full_range, 0.0)

    return df


def _microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market microstructure features from OHLCV data.

    These capture institutional vs retail behavior patterns.
    """
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]

    # ── Trade Intensity: volume per unit price move ──
    # High intensity = lots of volume for small move (absorption/institutional)
    price_move = (c - o).abs()
    df["trade_intensity"] = v / price_move.replace(0, np.nan)
    # Normalize to rolling z-score
    ti = df["trade_intensity"]
    ti_mean = ti.rolling(20, min_periods=1).mean()
    ti_std = ti.rolling(20, min_periods=1).std().replace(0, 1)
    df["trade_intensity"] = (ti - ti_mean) / ti_std

    # ── Pin Bar Detection (quantified) ──
    # Pin bar = wick > 60% of range on one side, body < 30% of range
    body_pct = (c - o).abs() / (h - l).replace(0, np.nan)
    upper_pct = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - l).replace(0, np.nan)
    lower_pct = (pd.concat([c, o], axis=1).min(axis=1) - l) / (h - l).replace(0, np.nan)

    # Bullish pin: long lower wick, body < 30%
    bull_pin = (lower_pct > 0.6) & (body_pct < 0.3)
    # Bearish pin: long upper wick, body < 30%
    bear_pin = (upper_pct > 0.6) & (body_pct < 0.3)
    df["pin_bar_score"] = np.where(bull_pin, lower_pct, np.where(bear_pin, -upper_pct, 0))

    # ── Gap Detection (crypto "gaps" = significant open vs prev close) ──
    gap = (o - c.shift(1)) / c.shift(1).replace(0, np.nan) * 100
    df["gap_pct"] = gap

    # ── Intrabar Volatility Ratio ──
    # How much price moved relative to the body direction
    # High ratio = lots of intrabar reversals (choppy)
    total_move = h - l
    directional_move = (c - o).abs()
    df["intrabar_vol_ratio"] = total_move / directional_move.replace(0, np.nan)

    # ── Relative Close Position (where did it close in the bar range) ──
    # 0 = at low, 0.5 = middle, 1 = at high
    # Rolling average tells about persistent buying/selling pressure
    close_pos = (c - l) / (h - l).replace(0, np.nan)
    df["close_position_avg"] = close_pos.rolling(5).mean()

    return df


def _statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced statistical features: Hurst exponent, auto-correlation,
    DI+/DI- ratio, fractal dimension proxy."""
    c, h, l = df["close"], df["high"], df["low"]
    ret = c.pct_change()

    # ── Hurst Exponent (simplified rescaled range method) ──
    # H > 0.5 = trending (persistent), H < 0.5 = mean-reverting, H = 0.5 = random
    window = 100
    hurst = pd.Series(0.5, index=df.index)
    for i in range(window, len(df)):
        chunk = ret.iloc[i - window:i].dropna().values
        if len(chunk) < 20:
            continue
        mean_r = chunk.mean()
        deviations = np.cumsum(chunk - mean_r)
        R = deviations.max() - deviations.min()
        S = chunk.std()
        if S > 0 and R > 0:
            hurst.iloc[i] = np.log(R / S) / np.log(window)
    df["hurst_exponent"] = hurst

    # ── Auto-correlation (lag-1 return serial correlation) ──
    # Vectorized: corr(ret, ret.shift(1)) over rolling window
    ret_lag1 = ret.shift(1)
    rolling_cov = ret.rolling(20).cov(ret_lag1)
    rolling_var = ret.rolling(20).var().replace(0, np.nan)
    df["autocorr_1"] = (rolling_cov / rolling_var).fillna(0)

    # ── Auto-correlation lag-5 ──
    ret_lag5 = ret.shift(5)
    rolling_cov5 = ret.rolling(30).cov(ret_lag5)
    rolling_var5 = ret.rolling(30).var().replace(0, np.nan)
    df["autocorr_5"] = (rolling_cov5 / rolling_var5).fillna(0)

    # ── DI+/DI- ratio (directional movement components) ──
    # We compute ADX but the ratio of DI+ to DI- tells direction
    high_diff = h.diff()
    low_diff = -l.diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    atr_col = df.get("ATR", pd.Series(1.0, index=df.index))
    plus_di = pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr_col.replace(0, np.nan)
    minus_di = pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr_col.replace(0, np.nan)

    # DI ratio: >1 = bullish dominance, <1 = bearish dominance
    df["di_ratio"] = plus_di / minus_di.replace(0, np.nan)

    # DI spread: normalized difference
    df["di_spread"] = (plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)

    # ── Return Variance Ratio (Lo-MacKinlay) ──
    # Tests for random walk. VR(q) ≈ 1 if random, >1 if trending, <1 if mean-reverting
    q = 5
    var_1 = ret.rolling(20).var()
    ret_q = c.pct_change(q)
    var_q = ret_q.rolling(20).var()
    df["variance_ratio"] = (var_q / (q * var_1)).replace([np.inf, -np.inf], np.nan)

    return df


def _whale_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Detect whale/smart money activity from volume anomalies.

    Whales leave footprints: massive volume on single candles, volume
    clusters near key levels, and absorption patterns (high volume, small body).
    """
    v = df["volume"]
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]

    # ── Whale Volume Score ──
    # Z-score of volume (how many std devs above normal)
    vol_mean = v.rolling(50, min_periods=10).mean()
    vol_std = v.rolling(50, min_periods=10).std().replace(0, 1)
    vol_zscore = (v - vol_mean) / vol_std

    # Whale candle = volume > 3 std AND body is small relative to range (absorption)
    body_pct = (c - o).abs() / (h - l).replace(0, np.nan)
    absorption = (vol_zscore > 2) & (body_pct < 0.3)

    # Score: 0 = normal, high = whale-like activity
    df["whale_score"] = np.where(
        absorption,
        vol_zscore * (1 - body_pct),  # high volume + small body = strong whale signal
        np.where(vol_zscore > 3, vol_zscore * 0.5, 0),  # just high volume = weaker
    )

    # ── Smart Money Divergence ──
    # Price makes new low but volume is declining (smart money not selling)
    # Or price makes new high but volume is declining (distribution)
    price_new_low = c == c.rolling(20).min()
    price_new_high = c == c.rolling(20).max()
    vol_declining = v < v.rolling(10).mean()

    df["smart_money_div"] = np.where(
        price_new_low & vol_declining, 1.0,   # bullish: low on declining volume
        np.where(price_new_high & vol_declining, -1.0, 0.0),  # bearish: high on declining vol
    )

    return df


def _entropy_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Shannon entropy proxy - measures market uncertainty.

    Fast vectorized: uses rolling std / rolling mean-abs-deviation ratio
    as entropy proxy. High ratio = more uniform distribution = higher entropy.
    """
    ret = df["close"].pct_change().fillna(0)
    # Entropy proxy: normalized range of returns in rolling window
    # More spread out returns = higher entropy
    roll_std = ret.rolling(20).std()
    roll_mad = ret.abs().rolling(20).mean().replace(0, np.nan)
    # For Gaussian: std/mad ≈ 1.25. Deviation from this = non-Gaussian = different entropy
    df["price_entropy"] = (roll_std / roll_mad).fillna(1.0)
    return df


def compute_relative_strength(
    pair_df: pd.DataFrame, btc_df: pd.DataFrame, window: int = 20,
) -> dict:
    """Compute relative strength of a pair vs BTC.

    Returns RS ratio, RS momentum, and classification.
    """
    if len(pair_df) < window or len(btc_df) < window:
        return {"rs_ratio": 1.0, "rs_momentum": 0, "classification": "neutral"}

    pair_ret = pair_df["close"].pct_change().tail(window)
    btc_ret = btc_df["close"].pct_change().tail(window)

    # Align lengths
    min_len = min(len(pair_ret), len(btc_ret))
    pair_ret = pair_ret.tail(min_len).values
    btc_ret = btc_ret.tail(min_len).values

    # Cumulative returns
    pair_cum = float(np.prod(1 + pair_ret) - 1) * 100
    btc_cum = float(np.prod(1 + btc_ret) - 1) * 100

    # RS ratio: pair performance / BTC performance
    rs_ratio = pair_cum - btc_cum  # relative outperformance in %

    # RS over last 5 vs last 20 (momentum of RS)
    if min_len >= 5:
        pair_short = float(np.prod(1 + pair_ret[-5:]) - 1) * 100
        btc_short = float(np.prod(1 + btc_ret[-5:]) - 1) * 100
        rs_short = pair_short - btc_short
        rs_momentum = rs_short - rs_ratio / 4  # acceleration
    else:
        rs_momentum = 0

    if rs_ratio > 3:
        classification = "strong_outperform"
    elif rs_ratio > 0:
        classification = "outperform"
    elif rs_ratio > -3:
        classification = "underperform"
    else:
        classification = "strong_underperform"

    return {
        "rs_ratio": round(rs_ratio, 2),
        "rs_momentum": round(rs_momentum, 2),
        "pair_return": round(pair_cum, 2),
        "btc_return": round(btc_cum, 2),
        "classification": classification,
    }


def detect_trend_lines(df: pd.DataFrame, lookback: int = 100) -> list[dict]:
    """Auto-detect trend lines by connecting swing highs and swing lows.

    Returns lines as [{start_idx, start_price, end_idx, end_price, type, slope}].
    """
    if len(df) < lookback:
        return []

    chunk = df.tail(lookback)
    highs = chunk["high"].values
    lows = chunk["low"].values
    n = len(chunk)

    # Find swing points (local extremes over 5-bar window)
    swing_highs = []
    swing_lows = []
    half = 5
    for i in range(half, n - half):
        if highs[i] == max(highs[i - half:i + half + 1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] == min(lows[i - half:i + half + 1]):
            swing_lows.append((i, float(lows[i])))

    lines = []

    # Resistance line: connect 2 most recent swing highs
    if len(swing_highs) >= 2:
        sh = swing_highs[-2:]
        slope = (sh[1][1] - sh[0][1]) / max(1, sh[1][0] - sh[0][0])
        # Extend to current bar
        end_price = sh[1][1] + slope * (n - 1 - sh[1][0])
        lines.append({
            "start_idx": sh[0][0],
            "start_price": round(sh[0][1], 6),
            "end_idx": n - 1,
            "end_price": round(end_price, 6),
            "type": "resistance",
            "slope_per_bar": round(slope, 6),
        })

    # Support line: connect 2 most recent swing lows
    if len(swing_lows) >= 2:
        sl = swing_lows[-2:]
        slope = (sl[1][1] - sl[0][1]) / max(1, sl[1][0] - sl[0][0])
        end_price = sl[1][1] + slope * (n - 1 - sl[1][0])
        lines.append({
            "start_idx": sl[0][0],
            "start_price": round(sl[0][1], 6),
            "end_idx": n - 1,
            "end_price": round(end_price, 6),
            "type": "support",
            "slope_per_bar": round(slope, 6),
        })

    return lines


def _adaptive_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adaptive Moving Averages: KAMA, DEMA, TEMA.

    These adapt to market conditions better than fixed-period MAs.
    Features are normalized as distance from price.
    """
    c = df["close"]

    # ── KAMA (Kaufman Adaptive MA) ──
    # Uses efficiency ratio to adapt smoothing constant
    n = 10
    fast_sc = 2 / (2 + 1)    # fast EMA constant (period 2)
    slow_sc = 2 / (30 + 1)   # slow EMA constant (period 30)

    er = df.get("efficiency_ratio", pd.Series(0.5, index=df.index))
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(np.nan, index=df.index)
    kama.iloc[n - 1] = float(c.iloc[n - 1])
    for i in range(n, len(df)):
        if np.isnan(kama.iloc[i - 1]):
            kama.iloc[i] = float(c.iloc[i])
        else:
            kama.iloc[i] = float(kama.iloc[i - 1] + sc.iloc[i] * (c.iloc[i] - kama.iloc[i - 1]))

    df["kama_dist"] = (c - kama) / c.replace(0, np.nan) * 100

    # ── DEMA (Double Exponential MA) ──
    # DEMA = 2 * EMA(n) - EMA(EMA(n))
    ema20 = c.ewm(span=20, adjust=False).mean()
    dema = 2 * ema20 - ema20.ewm(span=20, adjust=False).mean()
    df["dema_dist"] = (c - dema) / c.replace(0, np.nan) * 100

    # ── TEMA (Triple Exponential MA) ──
    # TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    ema1 = c.ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    ema3 = ema2.ewm(span=20, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    df["tema_dist"] = (c - tema) / c.replace(0, np.nan) * 100

    # ── KAMA Slope (trend direction from adaptive MA) ──
    df["kama_slope"] = kama.diff(3) / c.replace(0, np.nan) * 100

    return df


def _seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonality: how current hour/day compares to historical pattern.

    Uses historical average return for the same hour-of-day and day-of-week
    to create a seasonal bias feature.
    """
    if "time" not in df.columns:
        df["seasonal_hour_bias"] = 0.0
        df["seasonal_dow_bias"] = 0.0
        return df

    try:
        ts = pd.to_datetime(df["time"])
        ret = df["close"].pct_change() * 100

        # Hour-of-day seasonality
        hour = ts.dt.hour
        hour_means = ret.groupby(hour).transform("mean")
        hour_std = ret.rolling(50, min_periods=10).std().replace(0, 1)
        df["seasonal_hour_bias"] = hour_means / hour_std

        # Day-of-week seasonality
        dow = ts.dt.dayofweek
        dow_means = ret.groupby(dow).transform("mean")
        df["seasonal_dow_bias"] = dow_means / hour_std
    except Exception:
        df["seasonal_hour_bias"] = 0.0
        df["seasonal_dow_bias"] = 0.0

    return df


def compute_seasonality_profile(df: pd.DataFrame) -> dict:
    """Compute hourly and daily return profiles for visualization."""
    if "time" not in df.columns or len(df) < 100:
        return {"hourly": {}, "daily": {}}

    ts = pd.to_datetime(df["time"])
    ret = df["close"].pct_change() * 100

    # Hourly profile
    hour = ts.dt.hour
    hourly = {}
    for h in range(24):
        mask = hour == h
        vals = ret[mask].dropna()
        if len(vals) > 5:
            hourly[h] = {
                "avg_return": round(float(vals.mean()), 4),
                "win_rate": round(float((vals > 0).mean()) * 100, 1),
                "count": len(vals),
            }

    # Daily profile
    dow = ts.dt.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = {}
    for d in range(7):
        mask = dow == d
        vals = ret[mask].dropna()
        if len(vals) > 5:
            daily[day_names[d]] = {
                "avg_return": round(float(vals.mean()), 4),
                "win_rate": round(float((vals > 0).mean()) * 100, 1),
                "count": len(vals),
            }

    return {"hourly": hourly, "daily": daily}


def _cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dominant cycle detection using FFT and autocorrelation.

    Identifies the strongest repeating cycle in price data
    and computes where we are in that cycle (phase).
    """
    c = df["close"]
    ret = c.pct_change().fillna(0)

    cycle_phase = pd.Series(0.0, index=df.index)
    cycle_strength = pd.Series(0.0, index=df.index)
    dominant_period = pd.Series(0.0, index=df.index)

    window = 64  # must be power of 2 for efficient FFT

    for i in range(window, len(df)):
        chunk = ret.iloc[i - window:i].values

        # FFT
        fft = np.fft.rfft(chunk)
        magnitudes = np.abs(fft)

        # Skip DC component (index 0) and Nyquist
        if len(magnitudes) < 3:
            continue

        mags = magnitudes[1:-1]
        freqs = np.fft.rfftfreq(window)[1:-1]

        if len(mags) == 0 or mags.max() == 0:
            continue

        # Dominant frequency (highest magnitude)
        peak_idx = int(np.argmax(mags))
        peak_freq = freqs[peak_idx]
        period = 1.0 / peak_freq if peak_freq > 0 else window

        # Cycle strength: peak magnitude vs total (0=no cycle, 1=perfect sine)
        strength = float(mags[peak_idx] / mags.sum()) if mags.sum() > 0 else 0

        # Phase: where are we in the cycle (0 to 2*pi)
        phase = float(np.angle(fft[peak_idx + 1]))  # +1 for DC offset
        # Normalize to -1 to +1 (sine of phase)
        phase_signal = float(np.sin(phase))

        cycle_phase.iloc[i] = phase_signal
        cycle_strength.iloc[i] = strength
        dominant_period.iloc[i] = min(period, window)

    df["cycle_phase"] = cycle_phase
    df["cycle_strength"] = cycle_strength
    df["dominant_period"] = dominant_period

    return df


def _information_theory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Information-theoretic features: transfer entropy proxy, mutual information.

    Fast vectorized implementation using correlation-based proxies.
    Transfer Entropy proxy: how much does lagged volume predict price?
    Mutual Information proxy: correlation between price and volume changes.
    """
    c = df["close"]
    v = df["volume"]
    ret = c.pct_change().fillna(0)
    vol_change = v.pct_change().fillna(0)

    # Transfer Entropy proxy: correlation of vol(t-1) with price(t)
    # High correlation = volume predicts future price (information flow)
    vol_lag1 = vol_change.shift(1)
    te_proxy = ret.rolling(30).corr(vol_lag1).abs()  # abs correlation as TE proxy
    df["transfer_entropy"] = te_proxy.fillna(0)

    # Mutual Information proxy: absolute correlation + squared correlation
    # Squared correlation captures non-linear dependence
    linear_mi = ret.rolling(30).corr(vol_change).abs()
    squared_mi = (ret ** 2).rolling(30).corr(vol_change ** 2).abs()
    df["mutual_info_pv"] = ((linear_mi.fillna(0) + squared_mi.fillna(0)) / 2)

    return df


def _complexity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate Entropy proxy and Sample Entropy - measure complexity.

    Fast vectorized implementation using statistical proxies instead of
    O(n^2) template matching. Captures the same information: regularity
    vs randomness of the price series.
    """
    ret = df["close"].pct_change().fillna(0)

    # Fast ApEn proxy: uses lag-1 and lag-2 autocorrelation (vectorized)
    ret_lag1 = ret.shift(1)
    ret_lag2 = ret.shift(2)
    ac1 = ret.rolling(30).cov(ret_lag1) / ret.rolling(30).var().replace(0, np.nan)
    ac2 = ret.rolling(30).cov(ret_lag2) / ret.rolling(30).var().replace(0, np.nan)
    regularity = (ac1.abs().fillna(0) + ac2.abs().fillna(0)) / 2
    df["approx_entropy"] = (1 - regularity).clip(0, 2)

    # Sample Entropy proxy (ratio variant - fast)
    ret_series = df["close"].pct_change()
    short_std = ret_series.rolling(10).std()
    long_std = ret_series.rolling(50, min_periods=10).std().replace(0, np.nan)
    # High ratio = short-term vol diverging from long-term = less predictable
    df["sample_entropy_proxy"] = (short_std / long_std).fillna(1.0)

    return df


def _dfa_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Detrended Fluctuation Analysis (DFA) exponent.

    DFA is the gold standard for measuring long-range correlations
    in non-stationary time series. Unlike Hurst, it's robust to
    trends and non-stationarity in the data.

    alpha < 0.5: anti-correlated (mean-reverting)
    alpha = 0.5: uncorrelated (random walk)
    alpha > 0.5: long-range correlated (trending)
    alpha = 1.0: 1/f noise (pink noise)
    alpha > 1.0: non-stationary, unbounded
    """
    c = df["close"]
    ret = c.pct_change().fillna(0).values

    window = 100
    dfa_vals = np.zeros(len(df))

    for i in range(window, len(df)):
        chunk = ret[i - window:i]
        n = len(chunk)

        # Integrate: cumulative sum of (x - mean)
        y = np.cumsum(chunk - chunk.mean())

        # DFA: compute fluctuation at multiple box sizes
        box_sizes = [4, 8, 16, 32]
        fluctuations = []

        for box in box_sizes:
            if box >= n:
                continue
            n_boxes = n // box
            if n_boxes < 2:
                continue

            f_sum = 0
            for b in range(n_boxes):
                segment = y[b * box:(b + 1) * box]
                # Linear detrend
                x_axis = np.arange(box)
                if len(segment) != box:
                    continue
                coeffs = np.polyfit(x_axis, segment, 1)
                trend = np.polyval(coeffs, x_axis)
                residual = segment - trend
                f_sum += np.sqrt(np.mean(residual ** 2))

            fluctuations.append((np.log(box), np.log(f_sum / n_boxes + 1e-10)))

        if len(fluctuations) >= 2:
            log_n = np.array([f[0] for f in fluctuations])
            log_f = np.array([f[1] for f in fluctuations])
            # DFA exponent = slope of log-log plot
            if np.std(log_n) > 0:
                alpha = float(np.polyfit(log_n, log_f, 1)[0])
                dfa_vals[i] = max(0, min(2, alpha))

    df["dfa_exponent"] = dfa_vals
    return df


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature interaction terms - help tree models find non-linear patterns.

    Key interactions between indicators that traders actually use together:
    - RSI × Volume (oversold + high volume = strong reversal signal)
    - MACD × ADX (signal + trend strength = quality filter)
    - Momentum × Regime (trending market amplifies momentum)
    """
    def _safe_col(name, default=0):
        if name in df.columns:
            return df[name].fillna(default)
        return pd.Series(default, index=df.index)

    # RSI × Volume Ratio: oversold on high volume = strong buy signal
    rsi = _safe_col("RSI", 50) / 100  # normalize to 0-1
    vol = _safe_col("volume_ratio", 1).clip(0, 5) / 5  # normalize
    df["ix_rsi_volume"] = (1 - rsi) * vol  # high when RSI low AND volume high

    # MACD × ADX: trend signal × trend strength
    macd_norm = _safe_col("MACD", 0)
    close = df["close"].replace(0, np.nan)
    macd_pct = (macd_norm / close * 100).fillna(0).clip(-5, 5) / 5  # normalize
    adx_norm = _safe_col("ADX", 25) / 100
    df["ix_macd_adx"] = macd_pct * adx_norm

    # Momentum × Efficiency (trending amplifies momentum)
    mom = _safe_col("momentum_quality", 0).clip(-5, 5) / 5
    er = _safe_col("efficiency_ratio", 0.5)
    df["ix_momentum_regime"] = mom * er

    # Whale × Direction (whale activity aligned with trend)
    whale = _safe_col("whale_score", 0).clip(0, 10) / 10
    ofi = _safe_col("ofi_14", 0).clip(-1, 1)
    df["ix_whale_flow"] = whale * ofi

    # Squeeze × Cycle (squeeze at cycle bottom/top = breakout timing)
    squeeze = 1 - _safe_col("squeeze_ratio", 1).clip(0, 2) / 2
    cycle_phase = _safe_col("cycle_phase", 0)
    df["ix_squeeze_cycle"] = squeeze * cycle_phase.abs()

    return df


def _jump_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Detect price jumps vs continuous moves.

    Jumps are sudden large moves that differ from normal diffusion.
    Uses Barndorff-Nielsen & Shephard (2006) bipower variation approach.

    jump_indicator: 1 = jump detected, 0 = continuous
    jump_magnitude: size of detected jump (% of price)
    continuous_vol: volatility excluding jumps (cleaner vol estimate)
    """
    c = df["close"]
    ret = c.pct_change().fillna(0)
    abs_ret = ret.abs()

    # Realized variance (sum of squared returns)
    window = 20
    rv = (ret ** 2).rolling(window).sum()

    # Bipower variation (product of consecutive absolute returns)
    # BV = (pi/2) * sum(|r_t| * |r_{t-1}|)
    bv = (np.pi / 2) * (abs_ret * abs_ret.shift(1)).rolling(window).sum()

    # Jump component: RV - BV (positive = jumps present)
    jump_var = (rv - bv).clip(lower=0)

    # Jump test statistic
    bv_safe = bv.replace(0, np.nan)
    jump_ratio = jump_var / bv_safe  # ratio of jump to continuous variance

    # Jump indicator: significant if jump_ratio > threshold
    df["jump_indicator"] = (jump_ratio > 0.5).astype(float)

    # Jump magnitude: max single-bar return in window when jump detected
    rolling_max_ret = abs_ret.rolling(window).max()
    df["jump_magnitude"] = np.where(
        df["jump_indicator"] > 0,
        rolling_max_ret * 100,  # as percentage
        0,
    )

    # Continuous volatility (BV-based, cleaner than RV)
    df["continuous_vol"] = np.sqrt(bv / window).fillna(0)

    return df


def _kalman_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kalman Filter: optimal Bayesian price estimate from noisy observations.

    The Kalman filter models price as a hidden state with process noise
    (real price movement) and observation noise (market microstructure).
    Provides a smoothed "true" price estimate and a prediction error signal.

    Features:
    - kalman_residual: price - kalman estimate (positive = overpriced)
    - kalman_gain: current filter gain (low = confident in estimate)
    """
    prices = df["close"].values.astype(float)
    n = len(prices)

    # State: [price, velocity]
    # Transition: price(t) = price(t-1) + velocity(t-1)
    #             velocity(t) = velocity(t-1)

    # Initialize
    x = np.array([prices[0], 0.0])  # [price, velocity]
    P = np.array([[1.0, 0.0], [0.0, 1.0]])  # covariance

    # Process noise (how much true price changes per step)
    Q = np.array([[0.001, 0.0], [0.0, 0.0001]])
    # Measurement noise (market microstructure noise)
    R = np.array([[0.01]])
    # Transition matrix
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    # Measurement matrix
    H = np.array([[1.0, 0.0]])

    kalman_prices = np.zeros(n)
    kalman_gains = np.zeros(n)
    kalman_residuals = np.zeros(n)

    for i in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        z = prices[i]
        y = z - (H @ x_pred)[0]  # innovation (residual)
        S = (H @ P_pred @ H.T + R)[0, 0]  # innovation covariance
        K = (P_pred @ H.T) / max(S, 1e-10)  # Kalman gain

        x = x_pred + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P_pred

        kalman_prices[i] = x[0]
        kalman_gains[i] = float(K[0, 0])

        # Normalize residual by price
        if prices[i] > 0:
            kalman_residuals[i] = y / prices[i] * 100  # as %

    df["kalman_residual"] = kalman_residuals
    df["kalman_gain"] = kalman_gains

    return df


def _final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Final precision features: GK vol, RVI, VW momentum, Fisher, ConnorsRSI."""
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]

    # ── Garman-Klass Volatility (best single OHLC estimator) ──
    # GK = 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    log_hl = np.log(h / l.replace(0, np.nan)) ** 2
    log_co = np.log(c / o.replace(0, np.nan)) ** 2
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df["garman_klass_vol"] = np.sqrt(gk_var.rolling(20).mean().clip(lower=0))

    # ── Relative Vigor Index (RVI) ──
    # Measures conviction: close-open relative to high-low range
    numerator = ((c - o) + 2 * (c.shift(1) - o.shift(1)) + 2 * (c.shift(2) - o.shift(2)) + (c.shift(3) - o.shift(3))) / 6
    denominator = ((h - l) + 2 * (h.shift(1) - l.shift(1)) + 2 * (h.shift(2) - l.shift(2)) + (h.shift(3) - l.shift(3))) / 6
    df["rvi"] = (numerator / denominator.replace(0, np.nan)).rolling(10).mean()

    # ── Volume-Weighted Momentum ──
    # Momentum where each bar's return is weighted by its relative volume
    ret = c.pct_change()
    vol_weight = v / v.rolling(20, min_periods=1).mean().replace(0, np.nan)
    weighted_ret = ret * vol_weight
    df["vw_momentum_10"] = weighted_ret.rolling(10).sum()
    df["vw_momentum_20"] = weighted_ret.rolling(20).sum()

    # ── Ehlers Fisher Transform ──
    # Converts price to near-Gaussian distribution for sharper turning points
    period = 10
    hl_mid = (h.rolling(period).max() + l.rolling(period).min()) / 2
    hl_range = (h.rolling(period).max() - l.rolling(period).min()).replace(0, np.nan)
    raw_val = 2 * ((c - hl_mid) / hl_range) - 1  # normalize to [-1, 1]
    raw_val = raw_val.clip(-0.999, 0.999)  # prevent log singularity
    # Fisher transform: 0.5 * ln((1+x)/(1-x))
    fisher = 0.5 * np.log((1 + raw_val) / (1 - raw_val))
    df["fisher_transform"] = fisher.ewm(span=5).mean()

    # ── Connors RSI (institutional grade) ──
    # CRSI = (RSI(3) + RSI_streak(2) + PercentRank(100)) / 3
    rsi_3 = _compute_rsi_series(c, 3)

    # Streak RSI: count consecutive up/down days, then RSI of streak
    streak = pd.Series(0, index=df.index)
    direction = np.sign(c.diff())
    groups = (direction != direction.shift()).cumsum()
    streak = direction * direction.groupby(groups).cumcount().add(1)
    rsi_streak = _compute_rsi_series(streak, 2)

    # Percent Rank: where today's return sits in last 100 returns
    single_ret = c.pct_change()
    pct_rank = single_ret.rolling(100).rank(pct=True) * 100

    df["connors_rsi"] = (rsi_3 + rsi_streak + pct_rank) / 3

    # ── Chande Momentum Oscillator (CMO) ──
    # Like RSI but symmetric and zero-lag: (sum_up - sum_down) / (sum_up + sum_down) * 100
    delta = c.diff()
    gain_sum = delta.where(delta > 0, 0).rolling(14).sum()
    loss_sum = (-delta.where(delta < 0, 0)).rolling(14).sum()
    total = (gain_sum + loss_sum).replace(0, np.nan)
    df["cmo"] = ((gain_sum - loss_sum) / total * 100).fillna(0)

    # ── Elder Ray (Bull Power + Bear Power) ──
    # Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = c.ewm(span=13, adjust=False).mean()
    bull_power = h - ema13
    bear_power = l - ema13
    # Normalize by price for cross-asset comparability
    df["elder_bull"] = bull_power / c.replace(0, np.nan) * 100
    df["elder_bear"] = bear_power / c.replace(0, np.nan) * 100

    # ── Aroon Oscillator ──
    # Measures how recently price made a high vs low within a window
    # Range -100 to +100. Positive = uptrend, Negative = downtrend
    period = 25
    aroon_up = h.rolling(period + 1).apply(lambda x: x.argmax() / period * 100, raw=True)
    aroon_down = l.rolling(period + 1).apply(lambda x: x.argmin() / period * 100, raw=True)
    df["aroon_osc"] = aroon_up - aroon_down

    # ── Mass Index ──
    # Detects reversals via range expansion then contraction ("reversal bulge")
    # When Mass Index rises above 27 then drops below 26.5 = reversal signal
    ema_range = (h - l).ewm(span=9, adjust=False).mean()
    double_ema = ema_range.ewm(span=9, adjust=False).mean()
    ratio = ema_range / double_ema.replace(0, np.nan)
    df["mass_index"] = ratio.rolling(25).sum()

    # ── Klinger Volume Oscillator ──
    # Combines price trend direction with volume flow
    trend = np.where(
        (h + l + c) > (h.shift(1) + l.shift(1) + c.shift(1)), 1, -1
    )
    dm = h - l  # day's magnitude
    cm = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(1, len(df)):
        if trend[i] == trend[i - 1]:
            cm.iloc[i] = cm.iloc[i - 1] + dm.iloc[i]
        else:
            cm.iloc[i] = dm.iloc[i - 1] + dm.iloc[i]
    vf = v * abs(2 * dm / cm.replace(0, np.nan) - 1) * np.sign(trend) * 100
    kvo = vf.ewm(span=34, adjust=False).mean() - vf.ewm(span=55, adjust=False).mean()
    # Normalize by volume for comparability
    vol_mean = v.rolling(50, min_periods=1).mean().replace(0, np.nan)
    df["klinger_osc"] = kvo / vol_mean

    # ── Vortex Indicator ──
    # VI+ and VI-: measures positive and negative trend movement
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    vm_plus = (h - l.shift(1)).abs()
    vm_minus = (l - h.shift(1)).abs()
    period_vi = 14
    vi_plus = vm_plus.rolling(period_vi).sum() / tr.rolling(period_vi).sum().replace(0, np.nan)
    vi_minus = vm_minus.rolling(period_vi).sum() / tr.rolling(period_vi).sum().replace(0, np.nan)
    # Vortex diff: positive = bullish trend, negative = bearish
    df["vortex_diff"] = vi_plus - vi_minus

    # ── Detrended Price Oscillator (DPO) ──
    # Removes trend to isolate price cycles. Complements FFT cycle detection.
    # DPO = Close - SMA(period/2 + 1 bars ago)
    dpo_period = 20
    sma_shifted = c.rolling(dpo_period).mean().shift(dpo_period // 2 + 1)
    df["dpo"] = (c - sma_shifted) / c.replace(0, np.nan) * 100  # normalized %

    # ── Ultimate Oscillator (Larry Williams) ──
    # Multi-timeframe buying pressure: weighted combo of 7/14/28 periods
    bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)  # buying pressure
    tr_uo = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    avg7 = bp.rolling(7).sum() / tr_uo.rolling(7).sum().replace(0, np.nan)
    avg14 = bp.rolling(14).sum() / tr_uo.rolling(14).sum().replace(0, np.nan)
    avg28 = bp.rolling(28).sum() / tr_uo.rolling(28).sum().replace(0, np.nan)
    df["ultimate_osc"] = (4 * avg7 + 2 * avg14 + avg28) / 7 * 100

    # ── Awesome Oscillator (Bill Williams) ──
    # Measures market momentum using midpoint: SMA5(mid) - SMA34(mid)
    midpoint = (h + l) / 2
    ao = midpoint.rolling(5).mean() - midpoint.rolling(34).mean()
    df["awesome_osc"] = ao / c.replace(0, np.nan) * 100  # normalized %

    # ── Acceleration/Deceleration (AC) ──
    # Rate of change of AO: catches momentum shifts before price
    ao_sma5 = ao.rolling(5).mean()
    df["accel_decel"] = (ao - ao_sma5) / c.replace(0, np.nan) * 100

    # ── Supertrend Direction (simplified) ──
    # ATR-based trend following: +1 = uptrend, -1 = downtrend
    atr_col = df.get("ATR", (h - l).rolling(14).mean())
    multiplier = 3.0
    upper_band = (h + l) / 2 + multiplier * atr_col
    lower_band = (h + l) / 2 - multiplier * atr_col
    supertrend = pd.Series(0.0, index=df.index)
    direction = 1
    for i in range(1, len(df)):
        if c.iloc[i] > upper_band.iloc[i - 1]:
            direction = 1
        elif c.iloc[i] < lower_band.iloc[i - 1]:
            direction = -1
        supertrend.iloc[i] = direction
    df["supertrend_dir"] = supertrend

    # ── Schaff Trend Cycle (STC) ──
    # Combines MACD speed with Stochastic smoothing. Range 0-100.
    # Faster trend detection than MACD alone.
    macd_line = c.ewm(span=23, adjust=False).mean() - c.ewm(span=50, adjust=False).mean()
    stc_period = 10
    ll_macd = macd_line.rolling(stc_period).min()
    hh_macd = macd_line.rolling(stc_period).max()
    stoch1 = ((macd_line - ll_macd) / (hh_macd - ll_macd).replace(0, np.nan) * 100).fillna(50)
    smooth1 = stoch1.ewm(span=3, adjust=False).mean()
    ll_s1 = smooth1.rolling(stc_period).min()
    hh_s1 = smooth1.rolling(stc_period).max()
    stoch2 = ((smooth1 - ll_s1) / (hh_s1 - ll_s1).replace(0, np.nan) * 100).fillna(50)
    df["schaff_tc"] = stoch2.ewm(span=3, adjust=False).mean()

    # ── Coppock Curve ──
    # Long-term momentum bottom signal. WMA of sum of ROC(14) + ROC(11).
    # Adapted from monthly (14/11/10) to intraday.
    roc_14 = c.pct_change(14) * 100
    roc_11 = c.pct_change(11) * 100
    coppock_raw = roc_14 + roc_11
    # Weighted moving average (10 period)
    weights_wma = np.arange(1, 11)
    df["coppock_curve"] = coppock_raw.rolling(10).apply(
        lambda x: np.dot(x, weights_wma) / weights_wma.sum() if len(x) == 10 else 0, raw=True
    )

    # ── Know Sure Thing (KST) ──
    # Martin Pring: weighted sum of 4 smoothed ROCs at different periods
    # Captures momentum consensus across multiple timeframes
    roc1 = c.pct_change(10).rolling(10).mean() * 100
    roc2 = c.pct_change(15).rolling(10).mean() * 100
    roc3 = c.pct_change(20).rolling(10).mean() * 100
    roc4 = c.pct_change(30).rolling(15).mean() * 100
    kst = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
    kst_signal = kst.rolling(9).mean()
    df["kst"] = kst - kst_signal  # histogram (like MACD hist)

    # ── Ease of Movement (EMV) ──
    # Relates price change to volume: how easily price moves
    # High EMV = price moving easily on low volume (strong trend)
    distance = ((h + l) / 2) - ((h.shift(1) + l.shift(1)) / 2)
    box_ratio = (v / 1e6) / (h - l).replace(0, np.nan)  # volume per unit range
    emv = distance / box_ratio.replace(0, np.nan)
    df["ease_of_movement"] = emv.rolling(14).mean()

    # ── Normalized ATR (NATR) ──
    # ATR as percentage of price - better for cross-asset comparison
    if "ATR" in df.columns:
        df["natr"] = df["ATR"] / c.replace(0, np.nan) * 100
    else:
        df["natr"] = (h - l).rolling(14).mean() / c.replace(0, np.nan) * 100

    # ── Feature #150: Trend Persistence Score ──
    # Composite: how persistent and strong is the current trend?
    ema_align = np.sign(c.ewm(span=9).mean() - c.ewm(span=21).mean())
    adx_val = df.get("ADX", pd.Series(25, index=df.index)).fillna(25) / 100
    hurst_val = df.get("hurst_exponent", pd.Series(0.5, index=df.index)).fillna(0.5)
    consec = df.get("consecutive_dir", pd.Series(0, index=df.index)).fillna(0)
    consec_norm = (consec / 5).clip(-1, 1)
    df["trend_persistence"] = (ema_align * adx_val + hurst_val - 0.5 + consec_norm * 0.3).clip(-1, 1)

    # ── Parabolic SAR Direction ──
    # +1 = SAR below price (uptrend), -1 = SAR above (downtrend)
    af_start, af_step, af_max = 0.02, 0.02, 0.20
    sar = np.zeros(len(df))
    direction_sar = np.ones(len(df))
    if len(df) > 2:
        sar[0] = float(l.iloc[0])
        ep = float(h.iloc[0])
        af = af_start
        trend = 1
        for i in range(1, len(df)):
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if trend == 1:
                if float(l.iloc[i]) < sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = float(l.iloc[i])
                    af = af_start
                else:
                    if float(h.iloc[i]) > ep:
                        ep = float(h.iloc[i])
                        af = min(af + af_step, af_max)
            else:
                if float(h.iloc[i]) > sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = float(h.iloc[i])
                    af = af_start
                else:
                    if float(l.iloc[i]) < ep:
                        ep = float(l.iloc[i])
                        af = min(af + af_step, af_max)
            direction_sar[i] = trend
    df["psar_dir"] = direction_sar

    # ── Chaikin Oscillator ──
    # EMA(3) - EMA(10) of Accumulation/Distribution Line
    ad_line = df.get("ad_line", pd.Series(0, index=df.index))
    if ad_line.sum() == 0:
        mf_mult = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        ad_line = (mf_mult.fillna(0) * v).cumsum()
    df["chaikin_osc"] = ad_line.ewm(span=3).mean() - ad_line.ewm(span=10).mean()
    # Normalize
    ad_std = df["chaikin_osc"].rolling(50, min_periods=1).std().replace(0, 1)
    df["chaikin_osc"] = df["chaikin_osc"] / ad_std

    # ── Donchian Channel Position ──
    # Turtle Trading: where price sits within 20-period high/low channel
    # 0 = at low (breakout down), 1 = at high (breakout up)
    dc_high = h.rolling(20).max()
    dc_low = l.rolling(20).min()
    dc_range = (dc_high - dc_low).replace(0, np.nan)
    df["donchian_pos"] = (c - dc_low) / dc_range

    # ── Stochastic RSI ──
    # Apply Stochastic formula to RSI instead of price - more sensitive
    rsi_col = df.get("RSI", pd.Series(50, index=df.index))
    stoch_period = 14
    rsi_low = rsi_col.rolling(stoch_period).min()
    rsi_high = rsi_col.rolling(stoch_period).max()
    rsi_range = (rsi_high - rsi_low).replace(0, np.nan)
    df["stoch_rsi"] = ((rsi_col - rsi_low) / rsi_range * 100).fillna(50)

    # ── TRIX ──
    # Triple-smoothed EMA rate of change - ultimate noise filter
    ema1_t = c.ewm(span=15, adjust=False).mean()
    ema2_t = ema1_t.ewm(span=15, adjust=False).mean()
    ema3_t = ema2_t.ewm(span=15, adjust=False).mean()
    df["trix"] = ema3_t.pct_change() * 10000  # basis points

    # ── Balance of Power ──
    # (Close - Open) / (High - Low): who controls the bar
    # +1 = full bull control, -1 = full bear control
    df["balance_of_power"] = ((c - o) / (h - l).replace(0, np.nan)).rolling(14).mean()

    # ── Percentage Price Oscillator (PPO) ──
    # Like MACD but as percentage - cross-asset comparable
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["ppo"] = ((ema12 - ema26) / ema26.replace(0, np.nan)) * 100
    ppo_signal = df["ppo"].ewm(span=9, adjust=False).mean()
    df["ppo_hist"] = df["ppo"] - ppo_signal

    # ── Price Distance from VWAP Bands ──
    # How many standard deviations from VWAP (like BB but volume-weighted)
    if "VWAP" in df.columns:
        vwap = df["VWAP"]
        vwap_std = (c - vwap).rolling(20).std().replace(0, np.nan)
        df["vwap_zscore"] = (c - vwap) / vwap_std
    else:
        df["vwap_zscore"] = 0.0

    # ── McGinley Dynamic ──
    # Self-adjusting MA that speeds up in downtrends, slows in uptrends
    # MD(t) = MD(t-1) + (Close - MD(t-1)) / (N * (Close/MD(t-1))^4)
    n_mg = 14
    md = pd.Series(np.nan, index=df.index)
    md.iloc[n_mg - 1] = float(c.iloc[:n_mg].mean())
    for i in range(n_mg, len(df)):
        prev = md.iloc[i - 1]
        if prev > 0 and not np.isnan(prev):
            ratio = float(c.iloc[i]) / prev
            md.iloc[i] = prev + (float(c.iloc[i]) - prev) / (n_mg * max(ratio ** 4, 0.01))
        else:
            md.iloc[i] = float(c.iloc[i])
    df["mcginley_dist"] = (c - md) / c.replace(0, np.nan) * 100

    # ── Heikin-Ashi Smoothed Features ──
    ha_close = (o + h + l + c) / 4
    ha_open = o.copy()
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_body = ha_close - ha_open
    ha_range = (h - l).replace(0, np.nan)
    # HA body ratio: 1 = full body (strong trend), 0 = doji
    df["ha_body_ratio"] = ha_body.abs() / ha_range
    # HA trend: consecutive HA bullish/bearish candles
    ha_dir = np.sign(ha_body)
    ha_groups = (ha_dir != ha_dir.shift()).cumsum()
    df["ha_trend_strength"] = ha_dir * ha_dir.groupby(ha_groups).cumcount().add(1) / 5

    # ── Price Acceleration ──
    # Second derivative: rate of change of momentum
    velocity = c.diff()           # first derivative (speed)
    acceleration = velocity.diff()  # second derivative
    df["price_acceleration"] = acceleration / c.replace(0, np.nan) * 10000  # bps

    # ── Waddah Attar Explosion ──
    # Crypto-popular: MACD momentum vs BB volatility expansion
    macd_diff = c.ewm(span=20, adjust=False).mean() - c.ewm(span=40, adjust=False).mean()
    macd_prev = macd_diff.shift(1)
    trend_force = (macd_diff - macd_prev).abs()
    bb_mid = c.rolling(20).mean()
    bb_std_val = c.rolling(20).std()
    explosion = bb_std_val * 2  # BB width as volatility threshold
    # Positive = momentum exceeds volatility (explosive move)
    df["waddah_explosion"] = (trend_force - explosion) / c.replace(0, np.nan) * 1000

    # ── Squeeze Momentum (LazyBear) ──
    # Linear regression value of price minus midline during BB squeeze
    midline = c.rolling(20).mean()
    deviation = c - midline
    # Linreg slope of deviation over 20 bars
    x_vals = np.arange(20, dtype=float)
    x_mean = x_vals.mean()
    x_var = ((x_vals - x_mean) ** 2).sum()
    squeeze_mom = deviation.rolling(20).apply(
        lambda y: np.sum((x_vals - x_mean) * (y - y.mean())) / x_var if len(y) == 20 else 0,
        raw=True,
    )
    df["squeeze_momentum"] = squeeze_mom / c.replace(0, np.nan) * 100

    # ── Choppiness-Adjusted Momentum ──
    # Momentum scaled by trend clarity: strong in trends, zero in chop
    raw_mom = c.pct_change(10) * 100
    chop = df.get("choppiness", pd.Series(50, index=df.index)).fillna(50)
    trend_factor = ((61.8 - chop) / 61.8).clip(0, 1)  # 0 at chop>61.8, 1 at chop=0
    df["chop_adj_momentum"] = raw_mom * trend_factor

    # ── Volume-Price Confirmation ──
    # Correlation between price direction and volume over 10 bars
    # High = volume confirms price moves, Low = divergence
    price_dir = np.sign(c.diff())
    vol_change = v.diff()
    df["vol_price_confirm"] = price_dir.rolling(10).corr(vol_change)

    # ── Ehlers Instantaneous Trendline ──
    # Zero-lag adaptive filter using DSP principles
    it = pd.Series(0.0, index=df.index)
    prices = c.values.astype(float)
    for i in range(7, len(prices)):
        it.iloc[i] = (
            (4 * prices[i] + 3 * prices[i-1] + 2 * prices[i-2] + prices[i-3]) / 10.0 * 0.5 +
            it.iloc[i-1] * 0.5
        )
    df["ehlers_it_dist"] = (c - it) / c.replace(0, np.nan) * 100

    # ── Range Intensity ──
    # What fraction of the 20-bar range was actively traded in last 5 bars
    range_20 = (h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan)
    range_5 = h.rolling(5).max() - l.rolling(5).min()
    df["range_intensity"] = range_5 / range_20

    # ── Median Price Deviation ──
    # Distance from rolling median (more robust than mean)
    median_20 = c.rolling(20).median()
    df["median_dev"] = (c - median_20) / c.replace(0, np.nan) * 100

    # ── Trend Angle ──
    # Arctangent of linear regression slope over 14 bars (in degrees)
    # Steep angle = strong trend, flat = no trend
    x_lr = np.arange(14, dtype=float)
    x_mean_lr = x_lr.mean()
    x_var_lr = ((x_lr - x_mean_lr) ** 2).sum()
    slope = c.rolling(14).apply(
        lambda y: np.sum((x_lr - x_mean_lr) * (y - y.mean())) / x_var_lr if len(y) == 14 else 0,
        raw=True,
    )
    # Normalize slope by price then convert to angle
    norm_slope = slope / c.replace(0, np.nan) * 100
    df["trend_angle"] = np.degrees(np.arctan(norm_slope))

    # ── Relative Volatility Index (RVI by Donald Dorsey) ──
    # RSI applied to 10-bar standard deviation instead of price
    # Measures if volatility is expanding (>50) or contracting (<50)
    std_10 = c.rolling(10).std()
    df["rvi_dorsey"] = _compute_rsi_series(std_10, 14)

    # ── Polarized Fractal Efficiency (PFE) ──
    # Measures how efficiently price moves: straight line / actual path
    # +100 = perfect uptrend, -100 = perfect downtrend, 0 = random
    pfe_period = 10
    net_move = c - c.shift(pfe_period)
    path_len = c.diff().abs().rolling(pfe_period).sum().replace(0, np.nan)
    pfe_raw = (net_move / path_len * 100).fillna(0)
    df["pfe"] = pfe_raw.ewm(span=5, adjust=False).mean()

    # ── Center of Gravity (Ehlers) ──
    # Leading oscillator: weighted price position within window
    cg_period = 10
    weights_cg = np.arange(cg_period, 0, -1, dtype=float)
    numer = c.rolling(cg_period).apply(lambda x: np.dot(x, weights_cg), raw=True)
    denom = c.rolling(cg_period).sum().replace(0, np.nan)
    df["center_of_gravity"] = -(numer / denom - (cg_period + 1) / 2)

    return df


def _compute_rsi_series(series, period):
    """Helper: compute RSI on any series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _ou_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ornstein-Uhlenbeck mean-reversion parameters from log prices.

    The OU process models price as: dX = theta*(mu - X)*dt + sigma*dW
    - theta: mean-reversion speed (higher = faster reversion)
    - mu: equilibrium/fair price
    - distance: how far current price is from equilibrium

    These are estimated via OLS regression: X(t) - X(t-1) = a + b*X(t-1) + e
    Then: theta = -b, mu = -a/b, sigma = std(residuals)
    """
    log_price = np.log(df["close"].replace(0, np.nan)).ffill()

    window = 60
    theta_vals = np.zeros(len(df))
    ou_dist_vals = np.zeros(len(df))

    for i in range(window, len(df)):
        x = log_price.iloc[i - window:i].values
        if len(x) < 20 or np.any(np.isnan(x)):
            continue

        dx = np.diff(x)         # X(t) - X(t-1)
        x_lag = x[:-1]          # X(t-1)

        # OLS: dx = a + b * x_lag
        n = len(dx)
        x_mean = x_lag.mean()
        dx_mean = dx.mean()
        cov_xdx = np.sum((x_lag - x_mean) * (dx - dx_mean))
        var_x = np.sum((x_lag - x_mean) ** 2)

        if var_x > 0:
            b = cov_xdx / var_x
            a = dx_mean - b * x_mean

            theta = -b  # mean-reversion speed
            if b < 0:  # only meaningful if mean-reverting
                mu = -a / b  # equilibrium log-price
                # Distance from equilibrium (in %)
                current_log = x[-1]
                ou_dist_vals[i] = (current_log - mu) * 100

            theta_vals[i] = max(0, min(1, theta))  # clip to [0, 1]

    df["ou_theta"] = theta_vals
    df["ou_distance"] = ou_dist_vals

    return df


def compute_cross_asset_features(pair_df: pd.DataFrame, btc_df: pd.DataFrame) -> dict:
    """Compute cross-asset features: BTC as leading indicator for altcoins.

    Returns feature values to inject into the pair's feature set.
    """
    if len(pair_df) < 30 or len(btc_df) < 30:
        return {}

    min_len = min(len(pair_df), len(btc_df))
    pair_ret = pair_df["close"].pct_change().tail(min_len).values
    btc_ret = btc_df["close"].pct_change().tail(min_len).values

    # BTC lead-lag: correlation of BTC(t) with pair(t+1)
    if len(btc_ret) > 5:
        lead_corr = float(np.corrcoef(btc_ret[:-1], pair_ret[1:])[0, 1])
    else:
        lead_corr = 0

    # BTC momentum (last 5 bars) as predictor for alt
    btc_mom_5 = float(np.sum(btc_ret[-5:])) * 100

    # BTC volatility regime
    btc_vol = float(np.std(btc_ret[-20:])) if len(btc_ret) >= 20 else 0

    # Relative beta: how much the pair moves per 1% BTC move
    if len(btc_ret) >= 20:
        cov = float(np.cov(btc_ret[-20:], pair_ret[-20:])[0][1])
        var_btc = float(np.var(btc_ret[-20:]))
        beta = cov / var_btc if var_btc > 0 else 1
    else:
        beta = 1

    return {
        "btc_lead_corr": round(lead_corr, 4),
        "btc_momentum_5": round(btc_mom_5, 4),
        "btc_volatility": round(btc_vol, 6),
        "beta_vs_btc": round(beta, 3),
    }


def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> list[dict]:
    """Auto-detect swing high/low and compute Fibonacci retracement levels.

    Returns levels as list of dicts for chart rendering.
    """
    if len(df) < lookback:
        return []

    chunk = df.tail(lookback)
    highs = chunk["high"].values
    lows = chunk["low"].values
    closes = chunk["close"].values

    swing_high_idx = int(np.argmax(highs))
    swing_low_idx = int(np.argmin(lows))
    swing_high = float(highs[swing_high_idx])
    swing_low = float(lows[swing_low_idx])

    if swing_high <= swing_low:
        return []

    current = float(closes[-1])
    diff = swing_high - swing_low

    # Determine trend: if swing low is more recent, uptrend retracement
    is_uptrend = swing_low_idx > swing_high_idx

    FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    FIB_NAMES = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]

    levels = []
    for ratio, name in zip(FIB_RATIOS, FIB_NAMES):
        if is_uptrend:
            # Retracing down from high
            price = swing_high - diff * ratio
        else:
            # Retracing up from low
            price = swing_low + diff * ratio

        levels.append({
            "ratio": ratio,
            "name": name,
            "price": round(price, 6),
            "distance_pct": round((price - current) / current * 100, 2),
        })

    return levels


# ---------------------------------------------------------------------------
# Support / Resistance detection
# ---------------------------------------------------------------------------

def detect_support_resistance(
    df: pd.DataFrame, window: int = 10, threshold_pct: float = 0.3,
) -> list[dict]:
    """Detect support and resistance levels by finding pivot highs/lows
    and clustering nearby levels together."""
    if len(df) < window * 2:
        return []

    highs = df["high"].values
    lows = df["low"].values
    pivots: list[dict] = []

    half = window // 2
    for i in range(half, len(df) - half):
        # Pivot high: local max in window
        if highs[i] == max(highs[i - half : i + half + 1]):
            pivots.append({"price": float(highs[i]), "type": "resistance", "idx": i})
        # Pivot low: local min in window
        if lows[i] == min(lows[i - half : i + half + 1]):
            pivots.append({"price": float(lows[i]), "type": "support", "idx": i})

    if not pivots:
        return []

    # Cluster nearby pivots (within threshold_pct of each other)
    pivots.sort(key=lambda x: x["price"])
    clusters: list[list[dict]] = [[pivots[0]]]
    for p in pivots[1:]:
        last_avg = np.mean([x["price"] for x in clusters[-1]])
        if abs(p["price"] - last_avg) / last_avg * 100 < threshold_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    current_price = float(df["close"].iloc[-1])
    results = []
    for cluster in clusters:
        avg_price = float(np.mean([x["price"] for x in cluster]))
        touches = len(cluster)
        # Determine type by majority
        supports = sum(1 for x in cluster if x["type"] == "support")
        level_type = "support" if supports > len(cluster) / 2 else "resistance"
        strength = min(5, touches)  # 1-5 scale
        results.append({
            "price": round(avg_price, 6),
            "type": level_type,
            "touches": touches,
            "strength": strength,
            "distance_pct": round((avg_price - current_price) / current_price * 100, 2),
        })

    # Sort by distance from current price
    results.sort(key=lambda x: abs(x["distance_pct"]))
    return results[:10]  # Top 10 nearest


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame) -> dict:
    """Detect volume and price anomalies using z-scores."""
    if len(df) < 50:
        return {"volume_anomaly": False, "price_anomaly": False,
                "volume_zscore": 0, "price_zscore": 0, "description": "Insufficient data"}

    vol = df["volume"]
    vol_mean = vol.rolling(50).mean().iloc[-1]
    vol_std = vol.rolling(50).std().iloc[-1]
    vol_z = float((vol.iloc[-1] - vol_mean) / vol_std) if vol_std > 0 else 0

    ret = df["close"].pct_change()
    ret_std = ret.rolling(50).std().iloc[-1]
    ret_z = float(abs(ret.iloc[-1]) / ret_std) if ret_std > 0 else 0

    descriptions = []
    if abs(vol_z) > 3:
        descriptions.append(f"Volume {vol_z:.1f}x normal")
    if ret_z > 3:
        descriptions.append(f"Price move {ret_z:.1f}x normal")

    return {
        "volume_anomaly": bool(abs(vol_z) > 3),
        "volume_zscore": round(vol_z, 2),
        "price_anomaly": bool(ret_z > 3),
        "price_zscore": round(ret_z, 2),
        "description": " | ".join(descriptions) if descriptions else "No anomalies",
    }
