"""Abstract Base Model for Time Series Forecasting."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from darts import TimeSeries

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract Base Class for all forecasting models."""

    _last_fitted_series: TimeSeries | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model name."""

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Default hyperparameters."""

    @property
    def supports_covariates(self) -> bool:
        """Whether the model supports external covariates."""
        return False

    @property
    def supports_rolling_forecast(self) -> bool:
        """Whether the model supports rolling window forecasting.

        Override to True in models that should use predict_rolling() instead of predict()
        for multi-step forecasts (e.g., ARIMA, ExponentialSmoothing).
        """
        return False

    @abstractmethod
    def build_model(self, **kwargs: Any) -> Any:
        """Create the internal Darts model."""

    @abstractmethod
    def fit(self, series: TimeSeries, **kwargs: Any) -> "BaseModel":
        """Train the model. Returns self for chaining."""

    @abstractmethod
    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        """Generate forecast for n steps."""

    def predict_rolling(self, n: int, **kwargs: Any) -> TimeSeries:
        """Generate n-step forecast using rolling window approach.

        Forecasts 1 step at a time, appends the prediction to the series,
        refits the model on the extended series, and repeats.

        This mimics backtest behavior and can produce more realistic multi-step
        forecasts for local statistical models like ARIMA and ExponentialSmoothing.

        Args:
            n: Number of steps to forecast
            **kwargs: Additional arguments passed to predict()

        Returns:
            TimeSeries containing all n rolling forecasts
        """
        if self._last_fitted_series is None:
            raise ValueError(
                "Model must be fitted first with fit(). "
                "Rolling forecast requires the last fitted series."
            )

        current_series = self._last_fitted_series
        predictions: List[float] = []
        time_indices: List[pd.Timestamp] = []

        logger.debug(f"Starting rolling forecast for {n} steps")

        for step in range(n):
            try:
                # Predict 1 step ahead
                pred = self.predict(n=1, **kwargs)
                pred_value = float(pred.values(copy=False)[0, 0])
                predictions.append(pred_value)
                time_indices.append(pred.time_index[0])

                # Extend series with predicted value
                current_series = current_series.append(pred)

                # Refit model on extended series
                self.fit(current_series)

            except Exception as exc:
                logger.error(f"Rolling forecast failed at step {step + 1}/{n}: {exc}")
                raise

        logger.debug(f"Rolling forecast complete: {len(predictions)} predictions")

        # Reconstruct TimeSeries from predictions using explicit DatetimeIndex
        return TimeSeries.from_times_and_values(
            times=pd.DatetimeIndex(time_indices),
            values=np.array(predictions).reshape(-1, 1),
        )

    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return getattr(self, "params", {})

    def set_params(self, **params: Any) -> "BaseModel":
        """Update model parameters. Call before fit()."""
        if not hasattr(self, "params"):
            self.params = {}
        self.params.update(params)
        return self

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: Union[float, int],
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> TimeSeries:
        """Walk-forward backtesting."""
        if not hasattr(self, "_model") or self._model is None:
            self._model = self.build_model()

        return self._model.historical_forecasts(
            series=series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            verbose=verbose,
            **kwargs,
        )

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.name}({params_str})"


__all__ = ["BaseModel"]
