"""ARIMA Model for Time Series Forecasting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from darts import TimeSeries
from darts.models import ARIMA

from .base import BaseModel


class ARIMAModel(BaseModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) Model.
    
    Args:
        p: AR order (autoregressive terms)
        d: Differencing order
        q: MA order (moving average terms)
        seasonal_order: Optional (P, D, Q, m) for SARIMA
    """

    @property
    def name(self) -> str:
        return "ARIMA"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"p": 1, "d": 1, "q": 1, "seasonal_order": None}

    def __init__(
        self,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs: Any,
    ) -> None:
        self.params: Dict[str, Any] = {
            "p": p,
            "d": d,
            "q": q,
            "seasonal_order": seasonal_order,
            **kwargs,
        }
        self._model: Optional[ARIMA] = None

    def _normalize_seasonal(self, seasonal_order: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Statsmodels expects a tuple; None should become (0, 0, 0, 0)."""
        if seasonal_order is None:
            return (0, 0, 0, 0)
        return seasonal_order

    def build_model(self, **kwargs: Any) -> ARIMA:
        build_params = {**self.params, **kwargs}
        p = build_params.pop("p")
        d = build_params.pop("d")
        q = build_params.pop("q")
        seasonal_order = self._normalize_seasonal(build_params.pop("seasonal_order", None))
        return ARIMA(p=p, d=d, q=q, seasonal_order=seasonal_order, **build_params)

    def fit(self, series: TimeSeries, **kwargs: Any) -> "ARIMAModel":
        self._model = self.build_model()
        self._model.fit(series)
        return self

    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        if self._model is None:
            raise ValueError("Model must be fitted first. Call fit().")
        return self._model.predict(n=n, **kwargs)

    def __repr__(self) -> str:
        p, d, q = self.params.get("p", 1), self.params.get("d", 1), self.params.get("q", 1)
        seasonal = self.params.get("seasonal_order")
        if seasonal:
            P, D, Q, m = seasonal
            return f"ARIMA({p},{d},{q})({P},{D},{Q})[{m}]"
        return f"ARIMA({p},{d},{q})"


__all__ = ["ARIMAModel"]
