"""Exponential Smoothing Model for Time Series Forecasting."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode

from .base import BaseModel

TrendType = Literal["add", "mul", None]
SeasonalType = Literal["add", "mul", None]

_TREND_MAP = {None: None, "add": ModelMode.ADDITIVE, "mul": ModelMode.MULTIPLICATIVE}
_SEASONAL_MAP = {None: SeasonalityMode.NONE, "add": SeasonalityMode.ADDITIVE, "mul": SeasonalityMode.MULTIPLICATIVE}


class ExponentialSmoothingModel(BaseModel):
    """
    Exponential Smoothing (Holt-Winters) Model.

    Args:
        trend: None, "add", or "mul"
        damped: Use damped trend
        seasonal: None, "add", or "mul"
        seasonal_periods: Periods per seasonal cycle (288 = daily for 5min data)
    """

    @property
    def name(self) -> str:
        return "ExponentialSmoothing"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"trend": None, "damped": False, "seasonal": None, "seasonal_periods": None}

    @property
    def supports_rolling_forecast(self) -> bool:
        return True

    def __init__(
        self,
        trend: TrendType = None,
        damped: bool = False,
        seasonal: SeasonalType = None,
        seasonal_periods: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Only require seasonal_periods if seasonal is explicitly set to a value other than None
        if seasonal is not None and seasonal != "None" and seasonal_periods is None:
            raise ValueError("seasonal_periods required when seasonal is set")

        self.params: Dict[str, Any] = {
            "trend": trend,
            "damped": damped,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            **kwargs,
        }
        self._model: Optional[ExponentialSmoothing] = None

    def build_model(self, **kwargs: Any) -> ExponentialSmoothing:
        build_params = {**self.params, **kwargs}
        trend_str = build_params.pop("trend")
        damped = build_params.pop("damped")
        seasonal_str = build_params.pop("seasonal")
        seasonal_periods = build_params.pop("seasonal_periods")

        return ExponentialSmoothing(
            trend=_TREND_MAP.get(trend_str),
            damped=damped,
            seasonal=_SEASONAL_MAP.get(seasonal_str, SeasonalityMode.NONE),
            seasonal_periods=seasonal_periods,
            **build_params,
        )

    def fit(self, series: TimeSeries, **kwargs: Any) -> "ExponentialSmoothingModel":
        self._model = self.build_model()
        self._model.fit(series)
        self._last_fitted_series = series
        return self

    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        if self._model is None:
            raise ValueError("Model must be fitted first. Call fit().")
        return self._model.predict(n=n, **kwargs)

    def __repr__(self) -> str:
        trend, damped = self.params.get("trend"), self.params.get("damped", False)
        seasonal, periods = self.params.get("seasonal"), self.params.get("seasonal_periods")
        
        e = "A"
        t = "N" if trend is None else ("Ad" if damped else "A") if trend == "add" else ("Md" if damped else "M")
        s = "N" if seasonal is None else "A" if seasonal == "add" else "M"
        
        notation = f"ETS({e},{t},{s})"
        return f"{notation}[{periods}]" if periods else notation


__all__ = ["ExponentialSmoothingModel"]
