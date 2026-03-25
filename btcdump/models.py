"""ML pipeline: walk-forward training, weighted ensemble, persistence.

This module fixes the critical data leakage bugs from the original code:
1. Indicators are computed per fold on training-visible data only.
2. Each TrainedEnsemble carries its own StandardScaler.
3. Walk-forward expanding-window validation replaces simple 85/15 split.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from btcdump import indicators
from btcdump.config import AppConfig
from btcdump.features import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result of one walk-forward fold."""

    fold_idx: int
    train_size: int
    test_size: int
    mape: Dict[str, float]
    ensemble_mape: float
    predictions: np.ndarray
    actuals: np.ndarray


@dataclass
class TrainedEnsemble:
    """All artifacts needed for prediction, stored together."""

    models: Dict[str, object]
    scaler: StandardScaler
    weights: Dict[str, float]
    fold_results: List[FoldResult]
    trained_at: str
    interval: str
    symbol: str
    train_candles: int
    config_hash: str
    feature_importances: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    @property
    def avg_mape(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.ensemble_mape for f in self.fold_results]))


class ModelPipeline:
    """Handles training, prediction, and persistence of ensemble models."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._feature_eng = FeatureEngineer(config.features)

    def train_walk_forward(
        self,
        raw_df: pd.DataFrame,
        symbol: str = "",
        interval: str = "",
        progress_callback: Optional[callable] = None,
    ) -> TrainedEnsemble:
        """Train with expanding-window walk-forward validation.

        Steps:
        1. Create expanding folds.
        2. Per fold: compute indicators on train only, train, evaluate on test.
        3. Compute inverse-MAPE ensemble weights.
        4. Final train on all data.
        """
        mc = self._config.model
        ic = self._config.indicators
        fold_results: List[FoldResult] = []

        total_len = len(raw_df)
        n_folds = mc.walk_forward_folds

        for fold_idx in range(n_folds):
            train_end = mc.min_train_size + fold_idx * mc.test_size
            test_end = train_end + mc.test_size

            if test_end > total_len:
                logger.warning("Fold %d: insufficient data, stopping", fold_idx)
                break

            # --- Train phase: indicators only on training data ---
            train_slice = raw_df.iloc[:train_end].copy()
            train_enriched = indicators.compute_all(train_slice, ic)
            X_train, y_train = self._feature_eng.build(train_enriched)

            if len(X_train) == 0:
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            models = self._create_models()
            for model in models.values():
                model.fit(X_train_scaled, y_train)

            # --- Test phase: indicators on train+test (proper history) ---
            full_slice = raw_df.iloc[:test_end].copy()
            full_enriched = indicators.compute_all(full_slice, ic)
            X_all, y_all = self._feature_eng.build(full_enriched)

            n_train_samples = len(X_train)
            if len(X_all) <= n_train_samples:
                continue

            X_test = X_all[n_train_samples:]
            y_test = y_all[n_train_samples:]

            if len(X_test) == 0:
                continue

            X_test_scaled = scaler.transform(X_test)

            model_mape: Dict[str, float] = {}
            model_preds: Dict[str, np.ndarray] = {}
            for name, model in models.items():
                pred = model.predict(X_test_scaled)
                model_preds[name] = pred
                model_mape[name] = float(mean_absolute_percentage_error(y_test, pred))

            ensemble_pred = np.mean(list(model_preds.values()), axis=0)
            ensemble_mape = float(mean_absolute_percentage_error(y_test, ensemble_pred))

            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                train_size=len(X_train),
                test_size=len(X_test),
                mape=model_mape,
                ensemble_mape=ensemble_mape,
                predictions=ensemble_pred,
                actuals=y_test,
            ))

            if progress_callback:
                progress_callback(fold_idx + 1, n_folds)

        # --- Compute weights from fold results ---
        weights = self._compute_weights(fold_results)

        # --- Final training on all data ---
        full_enriched = indicators.compute_all(raw_df.copy(), ic)
        X_final, y_final = self._feature_eng.build(full_enriched)

        if len(X_final) == 0:
            raise ValueError("Not enough data for final model training")

        final_scaler = StandardScaler()
        X_final_scaled = final_scaler.fit_transform(X_final)

        final_models = self._create_models()
        for model in final_models.values():
            model.fit(X_final_scaled, y_final)

        # --- Feature importance extraction ---
        feat_names = list(self._config.features.feature_columns)
        n_raw = len(feat_names)
        window = self._config.features.window_size
        agg = np.zeros(n_raw)
        for name, model in final_models.items():
            fi = model.feature_importances_  # shape: (n_raw * window,)
            collapsed = np.zeros(n_raw)
            for i in range(n_raw):
                collapsed[i] = fi[i * window : (i + 1) * window].sum()
            agg += collapsed * weights.get(name, 1 / 3)
        total = agg.sum()
        if total > 0:
            agg = agg / total

        return TrainedEnsemble(
            models=final_models,
            scaler=final_scaler,
            weights=weights,
            fold_results=fold_results,
            trained_at=datetime.now().isoformat(),
            interval=interval,
            symbol=symbol,
            train_candles=len(raw_df),
            config_hash=self._config_hash(),
            feature_importances=agg.tolist(),
            feature_names=feat_names,
        )

    def predict(
        self, ensemble: TrainedEnsemble, raw_df: pd.DataFrame,
    ) -> Tuple[float, float, Dict[str, float]]:
        """Generate weighted prediction with confidence score.

        Returns (prediction, confidence_pct, individual_predictions).
        """
        enriched = indicators.compute_all(raw_df.copy(), self._config.indicators)
        latest = self._feature_eng.build_latest(enriched)
        scaled = ensemble.scaler.transform(latest)

        preds: Dict[str, float] = {}
        for name, model in ensemble.models.items():
            preds[name] = float(model.predict(scaled)[0])

        # Weighted ensemble
        prediction = sum(preds[n] * ensemble.weights[n] for n in preds)

        # Confidence: inverse of coefficient of variation
        pred_arr = np.array(list(preds.values()))
        mean_pred = np.mean(pred_arr)
        if mean_pred != 0:
            cv = np.std(pred_arr) / abs(mean_pred)
        else:
            cv = 1.0
        confidence = max(0.0, min(100.0, 100.0 * (1.0 - cv / 0.02)))

        return prediction, confidence, preds

    # --- Persistence ---

    def save(self, ensemble: TrainedEnsemble, path: Optional[Path] = None) -> Path:
        """Save ensemble to disk."""
        if path is None:
            path = (
                self._config.model.models_dir
                / f"{ensemble.symbol}_{ensemble.interval}.joblib"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ensemble, path)
        logger.info("Model saved to %s", path)
        return path

    def load(self, symbol: str, interval: str) -> Optional[TrainedEnsemble]:
        """Load ensemble from disk. Returns None if not found."""
        path = self._config.model.models_dir / f"{symbol}_{interval}.joblib"
        if not path.exists():
            return None
        try:
            ensemble = joblib.load(path)
            if not isinstance(ensemble, TrainedEnsemble):
                logger.warning("Invalid model file: %s", path)
                return None
            if ensemble.config_hash != self._config_hash():
                logger.warning("Config changed since model was saved, ignoring cache")
                return None
            return ensemble
        except Exception:
            logger.exception("Failed to load model from %s", path)
            return None

    def should_retrain(
        self, ensemble: TrainedEnsemble, current_candle_count: int,
    ) -> bool:
        """Check if enough new candles have arrived to justify retraining."""
        new_candles = current_candle_count - ensemble.train_candles
        return new_candles >= self._config.model.retrain_interval_candles

    # --- Private helpers ---

    def _create_models(self) -> Dict[str, object]:
        mc = self._config.model
        return {
            "xgb": XGBRegressor(**mc.xgb_params),
            "rf": RandomForestRegressor(**mc.rf_params),
            "gb": GradientBoostingRegressor(**mc.gb_params),
        }

    def _compute_weights(self, fold_results: List[FoldResult]) -> Dict[str, float]:
        """Inverse-MAPE weights normalized to sum to 1."""
        if not fold_results:
            return {"xgb": 1 / 3, "rf": 1 / 3, "gb": 1 / 3}

        model_names = list(fold_results[0].mape.keys())
        avg_mape: Dict[str, float] = {}
        for name in model_names:
            avg_mape[name] = float(np.mean([fr.mape[name] for fr in fold_results]))

        inv = {n: 1.0 / (m + 1e-10) for n, m in avg_mape.items()}
        total = sum(inv.values())
        weights = {n: v / total for n, v in inv.items()}
        logger.info("Ensemble weights: %s", weights)
        return weights

    def _config_hash(self) -> str:
        """Hash of model+feature config for staleness detection."""
        blob = json.dumps({
            "features": list(self._config.features.feature_columns),
            "window": self._config.features.window_size,
            "xgb": self._config.model.xgb_params,
            "rf": self._config.model.rf_params,
            "gb": self._config.model.gb_params,
        }, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()
