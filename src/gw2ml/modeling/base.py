"""Abstract Base Model for Time Series Forecasting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from darts import TimeSeries


class BaseModel(ABC):
    """Abstract Base Class for all forecasting models."""

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

    @abstractmethod
    def build_model(self, **kwargs: Any) -> Any:
        """Create the internal Darts model."""

    @abstractmethod
    def fit(self, series: TimeSeries, **kwargs: Any) -> "BaseModel":
        """Train the model. Returns self for chaining."""

    @abstractmethod
    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        """Generate forecast for n steps."""

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
