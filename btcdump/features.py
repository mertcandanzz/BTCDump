"""Feature engineering: sliding-window feature matrices from indicator DataFrames.

This module does NOT compute indicators. The caller must provide an
already-enriched DataFrame. This ensures indicators are computed only
on training-visible data (preventing data leakage).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from btcdump.config import FeatureConfig


class FeatureEngineer:
    """Creates sliding-window feature matrices for ML models."""

    def __init__(self, config: FeatureConfig) -> None:
        self._config = config

    def build(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and target vector y.

        X[i] = flattened window of (window_size) rows x (n_features) columns.
        y[i] = close price at the NEXT row after the window.

        Returns (X, y) or (empty, empty) if insufficient data.
        """
        feature_cols = list(self._config.feature_columns)
        window = self._config.window_size

        # Drop NaN rows from indicator warmup
        df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

        if len(df_clean) < window + 1:
            return np.empty((0, self.n_features)), np.empty(0)

        X, y = [], []
        for i in range(window, len(df_clean) - 1):
            row = df_clean[feature_cols].iloc[i - window : i].values.flatten()
            X.append(row)
            y.append(df_clean["close"].iloc[i + 1])

        return np.array(X), np.array(y)

    def build_latest(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the most recent window as a single feature vector.

        Returns shape (1, window_size * n_features).
        """
        feature_cols = list(self._config.feature_columns)
        window = self._config.window_size

        df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

        if len(df_clean) < window:
            raise ValueError(
                f"Need at least {window} clean rows, got {len(df_clean)}"
            )

        latest = df_clean[feature_cols].iloc[-window:].values.flatten()
        return latest.reshape(1, -1)

    @property
    def n_features(self) -> int:
        return self._config.window_size * len(self._config.feature_columns)
