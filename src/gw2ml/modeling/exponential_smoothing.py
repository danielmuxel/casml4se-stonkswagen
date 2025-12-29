"""
Exponential Smoothing Model for Time Series Forecasting.

Exponential Smoothing is a family of forecasting methods that use weighted
averages of past observations, with weights decaying exponentially.

Variants:
- Simple (N,N): No trend, no seasonality
- Holt (A,N): Additive trend, no seasonality
- Holt-Winters (A,A): Additive trend, additive seasonality
- Damped variants: Trend that flattens over time

Example:
    >>> from gw2ml.modeling import ExponentialSmoothingModel
    >>> from gw2ml.data import load_gw2_series
    >>>
    >>> data = load_gw2_series(19697, days_back=30)
    >>> train, test = data.split(train=0.8)
    >>>
    >>> # Simple Exponential Smoothing
    >>> model = ExponentialSmoothingModel()
    >>> model.fit(train.series)
    >>> forecast = model.predict(n=len(test.series))
    >>>
    >>> # Holt-Winters with daily seasonality
    >>> model = ExponentialSmoothingModel(
    ...     trend="add",
    ...     seasonal="add",
    ...     seasonal_periods=288  # 288 * 5min = 1 day
    ... )
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode

from .base import BaseModel


# Type aliases for clarity
TrendType = Literal["add", "mul", None]
SeasonalType = Literal["add", "mul", None]

# Mapping from string to Darts enums
_TREND_MAP = {
    None: None,
    "add": ModelMode.ADDITIVE,
    "mul": ModelMode.MULTIPLICATIVE,
}

_SEASONAL_MAP = {
    None: SeasonalityMode.NONE,
    "add": SeasonalityMode.ADDITIVE,
    "mul": SeasonalityMode.MULTIPLICATIVE,
}


class ExponentialSmoothingModel(BaseModel):
    """
    Exponential Smoothing Model for time series forecasting.

    This model wraps Darts' ExponentialSmoothing implementation and provides
    a consistent interface through BaseModel.

    Parameters:
        trend: Type of trend component.
            - None: No trend (Simple Exponential Smoothing)
            - "add": Additive trend (Holt's method)
            - "mul": Multiplicative trend
        damped: Whether to use damped trend (only if trend is not None).
            Damped trends flatten out over time, often more realistic.
        seasonal: Type of seasonal component.
            - None: No seasonality
            - "add": Additive seasonality (Holt-Winters additive)
            - "mul": Multiplicative seasonality (Holt-Winters multiplicative)
        seasonal_periods: Number of periods in a complete seasonal cycle.
            For 5-minute data: 288 = daily, 2016 = weekly

    Common Configurations:
        - Simple ES: trend=None, seasonal=None
        - Holt: trend="add", seasonal=None
        - Holt-Winters Additive: trend="add", seasonal="add"
        - Holt-Winters Multiplicative: trend="mul", seasonal="mul"
        - Damped Holt: trend="add", damped=True, seasonal=None

    Example:
        >>> # Simple Exponential Smoothing
        >>> model = ExponentialSmoothingModel()
        >>>
        >>> # Holt's Linear Trend
        >>> model = ExponentialSmoothingModel(trend="add")
        >>>
        >>> # Holt-Winters with daily seasonality
        >>> model = ExponentialSmoothingModel(
        ...     trend="add",
        ...     seasonal="add",
        ...     seasonal_periods=288
        ... )
        >>>
        >>> # Damped trend (more conservative forecasts)
        >>> model = ExponentialSmoothingModel(trend="add", damped=True)

    Note:
        ExponentialSmoothing is a LocalForecastingModel in Darts, which means:
        - It requires retrain=True for historical_forecasts()
        - It's trained on a single series
        - It's computationally efficient
    """

    @property
    def name(self) -> str:
        """Model name for logging and identification."""
        return "ExponentialSmoothing"

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default Exponential Smoothing parameters."""
        return {
            "trend": None,
            "damped": False,
            "seasonal": None,
            "seasonal_periods": None,
        }

    def __init__(
        self,
        trend: TrendType = None,
        damped: bool = False,
        seasonal: SeasonalType = None,
        seasonal_periods: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Exponential Smoothing model.

        Args:
            trend: Type of trend component.
                - None: No trend
                - "add": Additive trend
                - "mul": Multiplicative trend
            damped: Use damped trend (flattens over time).
            seasonal: Type of seasonal component.
                - None: No seasonality
                - "add": Additive seasonality
                - "mul": Multiplicative seasonality
            seasonal_periods: Number of periods in seasonal cycle.
                For 5-min data: 288 (daily), 2016 (weekly)
            **kwargs: Additional arguments passed to Darts ExponentialSmoothing.
        """
        # Validate seasonal_periods
        if seasonal is not None and seasonal_periods is None:
            raise ValueError(
                "seasonal_periods must be specified when seasonal is not None"
            )

        self.params: Dict[str, Any] = {
            "trend": trend,
            "damped": damped,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            **kwargs,
        }
        self._model: Optional[ExponentialSmoothing] = None

    def build_model(self, **kwargs: Any) -> ExponentialSmoothing:
        """
        Build the Darts ExponentialSmoothing model with current parameters.

        Args:
            **kwargs: Override parameters for this build.

        Returns:
            Configured Darts ExponentialSmoothing model.
        """
        # Merge stored params with any overrides
        build_params = {**self.params, **kwargs}

        # Extract ES-specific params and convert to Darts enums
        trend_str = build_params.pop("trend")
        damped = build_params.pop("damped")
        seasonal_str = build_params.pop("seasonal")
        seasonal_periods = build_params.pop("seasonal_periods")

        # Convert string params to Darts enums
        trend = _TREND_MAP.get(trend_str, None)
        seasonal = _SEASONAL_MAP.get(seasonal_str, SeasonalityMode.NONE)

        return ExponentialSmoothing(
            trend=trend,
            damped=damped,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            **build_params,
        )

    def fit(self, series: TimeSeries, **kwargs: Any) -> "ExponentialSmoothingModel":
        """
        Fit the Exponential Smoothing model on the training series.

        Args:
            series: Training TimeSeries.
            **kwargs: Additional arguments for fit.

        Returns:
            self (for method chaining).

        Example:
            >>> model = ExponentialSmoothingModel(trend="add")
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
            >>> forecast = model.predict(n=12)   # 12 steps = 1 hour
            >>> forecast = model.predict(n=288)  # 288 steps = 1 day
        """
        if self._model is None:
            raise ValueError(
                "Model must be fitted before predicting. Call fit() first."
            )
        return self._model.predict(n=n, **kwargs)

    def _get_model_notation(self) -> str:
        """
        Get ETS notation string (Error, Trend, Seasonal).

        Returns:
            String like "ETS(A,N,N)" or "ETS(A,Ad,A)"
        """
        trend = self.params.get("trend")
        damped = self.params.get("damped", False)
        seasonal = self.params.get("seasonal")

        # Error is always additive in statsmodels implementation
        e = "A"

        # Trend notation
        if trend is None:
            t = "N"
        elif trend == "add":
            t = "Ad" if damped else "A"
        else:  # mul
            t = "Md" if damped else "M"

        # Seasonal notation
        if seasonal is None:
            s = "N"
        elif seasonal == "add":
            s = "A"
        else:  # mul
            s = "M"

        return f"ETS({e},{t},{s})"

    def __repr__(self) -> str:
        """String representation with ETS notation."""
        notation = self._get_model_notation()
        seasonal_periods = self.params.get("seasonal_periods")

        if seasonal_periods:
            return f"{notation}[{seasonal_periods}]"
        return notation


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = ["ExponentialSmoothingModel"]

