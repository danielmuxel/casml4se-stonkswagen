"""
Abstract Base Model for Time Series Forecasting.

This module provides the BaseModel ABC that all forecasting models must inherit.
It ensures a consistent interface across ARIMA, ExponentialSmoothing, and custom models.

Example:
    >>> class MyModel(BaseModel):
    ...     name = "MyModel"
    ...     default_params = {"param1": 1}
    ...
    ...     def build_model(self, **kwargs):
    ...         return SomeDartsModel(**self.params)
    ...
    ...     def fit(self, series, **kwargs):
    ...         self._model = self.build_model()
    ...         self._model.fit(series)
    ...         return self
    ...
    ...     def predict(self, n, **kwargs):
    ...         return self._model.predict(n)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from darts import TimeSeries


# ══════════════════════════════════════════════════════════════════════════════
# BASE MODEL ABC
# ══════════════════════════════════════════════════════════════════════════════


class BaseModel(ABC):
    """
    Abstract Base Class for all forecasting models.

    This class defines the interface that all models must implement.
    It wraps Darts models and provides a consistent API for:
    - Training (fit)
    - Forecasting (predict)
    - Backtesting (historical_forecasts)

    Attributes:
        name: Unique model identifier (e.g., 'ARIMA', 'ExponentialSmoothing')
        default_params: Default hyperparameters for the model
        supports_covariates: Whether the model supports external covariates
        params: Current model parameters (set in __init__ of subclass)
        _model: Internal Darts model instance (set after build_model)

    Example:
        >>> model = ARIMAModel(p=2, d=1, q=2)
        >>> model.fit(train_series)
        >>> forecast = model.predict(n=12)
    """

    # ══════════════════════════════════════════════════════════════════════════
    # ABSTRACT PROPERTIES (must be overridden)
    # ══════════════════════════════════════════════════════════════════════════

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name of the model.

        This is used for logging, MLflow tracking, and model identification.

        Returns:
            Model name (e.g., 'ARIMA', 'ExponentialSmoothing')
        """

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """
        Default hyperparameters for the model.

        These are used when no parameters are provided to __init__.

        Returns:
            Dict of parameter names to default values
        """

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIONAL PROPERTIES (with defaults)
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def supports_covariates(self) -> bool:
        """
        Whether the model supports external covariates.

        Override this in subclasses that support past/future covariates.

        Returns:
            True if covariates are supported, False otherwise
        """
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS (must be implemented)
    # ══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def build_model(self, **kwargs: Any) -> Any:
        """
        Create and configure the internal Darts model.

        This method should instantiate the appropriate Darts model class
        with the parameters stored in self.params.

        Args:
            **kwargs: Additional model-specific parameters

        Returns:
            Configured Darts model instance
        """

    @abstractmethod
    def fit(self, series: TimeSeries, **kwargs: Any) -> "BaseModel":
        """
        Train the model on the given TimeSeries.

        Args:
            series: Training TimeSeries
            **kwargs: Additional training parameters (e.g., past_covariates)

        Returns:
            self (for method chaining)

        Example:
            >>> model.fit(train_series).predict(n=12)
        """

    @abstractmethod
    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        """
        Generate a forecast for n time steps.

        Args:
            n: Number of steps to forecast
            **kwargs: Additional prediction parameters

        Returns:
            TimeSeries with the forecasted values

        Raises:
            ValueError: If model has not been fitted
        """

    # ══════════════════════════════════════════════════════════════════════════
    # CONCRETE METHODS (standard implementations)
    # ══════════════════════════════════════════════════════════════════════════

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.

        Returns:
            Dict of parameter names to current values
        """
        return getattr(self, "params", {})

    def set_params(self, **params: Any) -> "BaseModel":
        """
        Update model parameters.

        Note: This should be called before fit(). Changing parameters
        after fitting requires re-fitting the model.

        Args:
            **params: Parameters to update

        Returns:
            self (for method chaining)
        """
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
        """
        Perform walk-forward backtesting.

        This method uses Darts' historical_forecasts() to generate
        forecasts at multiple points in time, simulating real-world usage.

        Args:
            series: Complete TimeSeries for backtesting
            start: Start point as ratio (0.0-1.0) or absolute index
            forecast_horizon: Number of steps per forecast
            stride: Steps between forecasts (default: 1)
            retrain: Whether to retrain at each step (default: False)
            verbose: Whether to print progress (default: False)
            **kwargs: Additional parameters for the model

        Returns:
            TimeSeries with all backtest forecasts concatenated

        Raises:
            ValueError: If model has not been built/fitted

        Example:
            >>> # Backtest starting from 80% of the series
            >>> forecasts = model.historical_forecasts(
            ...     series,
            ...     start=0.8,
            ...     forecast_horizon=12,
            ...     stride=12
            ... )
        """
        # Ensure model is built
        if not hasattr(self, "_model") or self._model is None:
            # Build and fit model if not already done
            self._model = self.build_model()

        # Delegate to Darts model's historical_forecasts
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
        """String representation of the model."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.name}({params_str})"


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = ["BaseModel"]

