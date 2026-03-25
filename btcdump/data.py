"""Data fetching, caching, and validation for Binance candle data."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from btcdump.config import DataConfig
from btcdump.utils import retry

logger = logging.getLogger(__name__)


@dataclass
class CandleData:
    """Validated candle DataFrame with metadata."""

    df: pd.DataFrame
    symbol: str
    interval: str
    fetched_at: datetime
    num_candles: int


class DataFetcher:
    """Fetches and caches OHLCV candle data from Binance."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "BTCDump/4.0"})
        # In-memory caches for exchange info and tickers
        self._exchange_info: Optional[List[Dict]] = None
        self._exchange_info_time: float = 0
        self._tickers: Optional[List[Dict]] = None
        self._tickers_time: float = 0

    @retry(max_retries=3, backoff=1.5, exceptions=(requests.RequestException,))
    def fetch(self, symbol: str, interval: str, limit: int = 0) -> CandleData:
        """Fetch candles from Binance public API."""
        if limit <= 0:
            limit = self._config.candle_limit

        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = self._session.get(
            self._config.base_url, params=params, timeout=self._config.request_timeout,
        )
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "ct", "qav", "trades", "tb_base", "tb_quote", "ignore",
        ])

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        df = self.validate(df)

        now = datetime.now()
        logger.info("Fetched %d candles for %s/%s", len(df), symbol, interval)

        return CandleData(
            df=df, symbol=symbol, interval=interval,
            fetched_at=now, num_candles=len(df),
        )

    def fetch_with_cache(
        self, symbol: str, interval: str, limit: int = 0,
    ) -> CandleData:
        """Return cached data if fresh; otherwise fetch from API."""
        cache_path = self._cache_path(symbol, interval)

        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self._config.cache_ttl_seconds:
                logger.info("Using cached data for %s/%s (age: %.0fs)", symbol, interval, age)
                return self._load_cache(cache_path, symbol, interval)

        data = self.fetch(symbol, interval, limit)
        self._save_cache(data, cache_path)
        return data

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Validate candle data integrity."""
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        # OHLC consistency: drop rows where high < low
        bad_mask = df["high"] < df["low"]
        if bad_mask.any():
            logger.warning("Dropping %d rows with high < low", bad_mask.sum())
            df = df[~bad_mask].reset_index(drop=True)

        # Volume must be non-negative
        df.loc[df["volume"] < 0, "volume"] = 0.0

        return df

    def _cache_path(self, symbol: str, interval: str) -> Path:
        self._config.cache_dir.mkdir(parents=True, exist_ok=True)
        return self._config.cache_dir / f"{symbol}_{interval}.csv"

    def _save_cache(self, data: CandleData, path: Path) -> None:
        data.df.to_csv(path, index=False)
        logger.info("Cached %d candles to %s", data.num_candles, path)

    def _load_cache(self, path: Path, symbol: str, interval: str) -> CandleData:
        df = pd.read_csv(path, parse_dates=["time"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return CandleData(
            df=df, symbol=symbol, interval=interval,
            fetched_at=datetime.fromtimestamp(path.stat().st_mtime),
            num_candles=len(df),
        )

    # ── Binance Discovery Endpoints ──────────────────────

    def fetch_exchange_info(self) -> List[Dict]:
        """Get all USDT trading pairs from Binance. Cached in memory."""
        now = time.time()
        if self._exchange_info and (now - self._exchange_info_time) < self._config.exchange_info_cache_ttl:
            return self._exchange_info

        url = f"{self._config.binance_api_base}/api/v3/exchangeInfo"
        resp = self._session.get(url, timeout=self._config.request_timeout)
        resp.raise_for_status()

        symbols = []
        for s in resp.json().get("symbols", []):
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                symbols.append({
                    "symbol": s["symbol"],
                    "baseAsset": s["baseAsset"],
                    "quoteAsset": s["quoteAsset"],
                })

        self._exchange_info = symbols
        self._exchange_info_time = now
        logger.info("Fetched %d USDT pairs from exchangeInfo", len(symbols))
        return symbols

    def fetch_tickers(self) -> List[Dict]:
        """Get 24h ticker stats for all USDT pairs. Cached in memory."""
        now = time.time()
        if self._tickers and (now - self._tickers_time) < self._config.ticker_cache_ttl:
            return self._tickers

        url = f"{self._config.binance_api_base}/api/v3/ticker/24hr"
        resp = self._session.get(url, timeout=self._config.request_timeout)
        resp.raise_for_status()

        tickers = []
        for t in resp.json():
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            tickers.append({
                "symbol": sym,
                "baseAsset": sym.replace("USDT", ""),
                "lastPrice": float(t.get("lastPrice", 0)),
                "priceChangePercent": float(t.get("priceChangePercent", 0)),
                "volume": float(t.get("volume", 0)),
                "quoteVolume": float(t.get("quoteVolume", 0)),
            })

        tickers.sort(key=lambda x: x["quoteVolume"], reverse=True)
        self._tickers = tickers
        self._tickers_time = now
        logger.info("Fetched %d USDT tickers", len(tickers))
        return tickers

    def fetch_mini_chart(
        self, symbol: str, interval: str = "1h", limit: int = 0,
    ) -> List[float]:
        """Get close prices only for sparkline charts. Lightweight."""
        if limit <= 0:
            limit = self._config.mini_chart_candles

        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = self._session.get(
            self._config.base_url, params=params, timeout=self._config.request_timeout,
        )
        resp.raise_for_status()
        return [float(candle[4]) for candle in resp.json()]  # index 4 = close
