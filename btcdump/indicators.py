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
    return df


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

    return df


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
