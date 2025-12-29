"""
ARIMA Model for Time Series Forecasting.

ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical
model for time series forecasting. It combines:
- AR (AutoRegressive): Uses past values to predict future values
- I (Integrated): Differencing to make the series stationary
- MA (Moving Average): Uses past forecast errors

Example:
    >>> from gw2ml.modeling import ARIMAModel
    >>> from gw2ml.data import load_gw2_series
    >>>
    >>> data = load_gw2_series(19697, days_back=30)
    >>> train, test = data.split(train=0.8)
    >>>
    >>> model = ARIMAModel(p=2, d=1, q=2)
    >>> model.fit(train.series)
    >>> forecast = model.predict(n=len(test.series))
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from darts import TimeSeries
from darts.models import ARIMA

from .base import BaseModel


class ARIMAModel(BaseModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) Model.

    This model wraps Darts' ARIMA implementation and provides a consistent
    interface through BaseModel.

    Parameters:
        p: AR order (number of autoregressive terms)
        d: Differencing order (number of times to difference)
        q: MA order (number of moving average terms)
        seasonal_order: Optional (P, D, Q, m) for SARIMA
            - P: Seasonal AR order
            - D: Seasonal differencing order
            - Q: Seasonal MA order
            - m: Seasonal period (e.g., 288 for daily with 5-min data)

    Attributes:
        name: "ARIMA"
        default_params: {"p": 1, "d": 1, "q": 1}

    Example:
        >>> # Simple ARIMA
        >>> model = ARIMAModel(p=2, d=1, q=2)
        >>>
        >>> # SARIMA with daily seasonality (288 * 5min = 1 day)
        >>> model = ARIMAModel(p=2, d=1, q=2, seasonal_order=(1, 1, 1, 288))

    Note:
        ARIMA is a LocalForecastingModel in Darts, which means:
        - It requires retrain=True for historical_forecasts()
        - It's trained on a single series
        - It's generally faster than deep learning models
    """

    @property
    def name(self) -> str:
        """Model name for logging and identification."""
        return "ARIMA"

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default ARIMA parameters."""
        return {
            "p": 1,
            "d": 1,
            "q": 1,
            "seasonal_order": None,
        }

    def __init__(
        self,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ARIMA model.

        Args:
            p: AR order (autoregressive). How many past values to use.
            d: Differencing order. How many times to difference the series.
            q: MA order (moving average). How many past errors to use.
            seasonal_order: Optional tuple (P, D, Q, m) for SARIMA.
                - P: Seasonal AR order
                - D: Seasonal differencing
                - Q: Seasonal MA order
                - m: Season length (e.g., 288 for daily with 5-min intervals)
            **kwargs: Additional arguments passed to Darts ARIMA.
        """
        self.params: Dict[str, Any] = {
            "p": p,
            "d": d,
            "q": q,
            "seasonal_order": seasonal_order,
            **kwargs,
        }
        self._model: Optional[ARIMA] = None

    def build_model(self, **kwargs: Any) -> ARIMA:
        """
        Build the Darts ARIMA model with current parameters.

        Args:
            **kwargs: Override parameters for this build.

        Returns:
            Configured Darts ARIMA model.
        """
        # Merge stored params with any overrides
        build_params = {**self.params, **kwargs}

        # Extract ARIMA-specific params
        p = build_params.pop("p")
        d = build_params.pop("d")
        q = build_params.pop("q")
        seasonal_order = build_params.pop("seasonal_order", None)

        return ARIMA(
            p=p,
            d=d,
            q=q,
            seasonal_order=seasonal_order,
            **build_params,
        )

    def fit(self, series: TimeSeries, **kwargs: Any) -> "ARIMAModel":
        """
        Fit the ARIMA model on the training series.

        Args:
            series: Training TimeSeries.
            **kwargs: Additional arguments for fit (not used by ARIMA).

        Returns:
            self (for method chaining).

        Example:
            >>> model = ARIMAModel(p=2, d=1, q=2)
            >>> model.fit(train_series)
            >>> forecast = model.predict(n=12)
        """
        self._model = self.build_model()
        self._model.fit(series)
        return self

    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        """
        Generate forecast for n time steps.

        Args:
            n: Number of steps to forecast.
            **kwargs: Additional arguments for predict.

        Returns:
            TimeSeries with forecasted values.

        Raises:
            ValueError: If model has not been fitted.

        Example:
            >>> forecast = model.predict(n=12)  # 12 steps = 1 hour
            >>> forecast = model.predict(n=288)  # 288 steps = 1 day
        """
        if self._model is None:
            raise ValueError(
                "Model must be fitted before predicting. Call fit() first."
            )
        return self._model.predict(n=n, **kwargs)

    def __repr__(self) -> str:
        """String representation with ARIMA notation."""
        p = self.params.get("p", 1)
        d = self.params.get("d", 1)
        q = self.params.get("q", 1)
        seasonal = self.params.get("seasonal_order")

        if seasonal:
            P, D, Q, m = seasonal
            return f"ARIMA({p},{d},{q})({P},{D},{Q})[{m}]"
        return f"ARIMA({p},{d},{q})"


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = ["ARIMAModel"]

