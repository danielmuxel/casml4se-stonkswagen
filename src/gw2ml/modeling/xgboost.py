"""XGBoost Model for Time Series Forecasting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from darts import TimeSeries
from darts.models import XGBModel

from .base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost Model using lag features for forecasting.
    
    Args:
        lags: Number of past values as features
        n_estimators: Number of boosting rounds
        max_depth: Max tree depth
        learning_rate: Step size shrinkage
        subsample: Row sampling ratio
        colsample_bytree: Column sampling ratio
    """

    @property
    def name(self) -> str:
        return "XGBoost"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "lags": 12, "output_chunk_length": 1, "n_estimators": 100,
            "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0,
        }

    def __init__(
        self,
        lags: Union[int, List[int]] = 12,
        output_chunk_length: int = 1,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.params: Dict[str, Any] = {
            "lags": lags, "output_chunk_length": output_chunk_length,
            "n_estimators": n_estimators, "max_depth": max_depth,
            "learning_rate": learning_rate, "subsample": subsample,
            "colsample_bytree": colsample_bytree, "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda, **kwargs,
        }
        self._model: Optional[XGBModel] = None

    def build_model(self, **kwargs: Any) -> XGBModel:
        return XGBModel(**{**self.params, **kwargs})

    def fit(self, series: TimeSeries, **kwargs: Any) -> "XGBoostModel":
        self._model = self.build_model()
        self._model.fit(series, **kwargs)
        return self

    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        if self._model is None:
            raise ValueError("Model must be fitted first. Call fit().")
        return self._model.predict(n=n, **kwargs)

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: float = 0.5,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: bool = False,
        **kwargs: Any,
    ) -> TimeSeries:
        """XGBoost can use retrain=False for faster backtesting."""
        if self._model is None:
            raise ValueError("Model must be fitted first. Call fit().")
        return self._model.historical_forecasts(
            series=series, start=start, forecast_horizon=forecast_horizon,
            stride=stride, retrain=retrain, **kwargs,
        )

    def __repr__(self) -> str:
        lags = self.params.get("lags", 12)
        n_est = self.params.get("n_estimators", 100)
        depth = self.params.get("max_depth", 5)
        lr = self.params.get("learning_rate", 0.1)
        return f"XGBoost(lags={lags}, n_est={n_est}, depth={depth}, lr={lr})"


__all__ = ["XGBoostModel"]
