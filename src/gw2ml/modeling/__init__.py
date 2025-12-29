"""
GW2ML Modeling Module.

This module provides the forecasting model framework:
- BaseModel: Abstract base class for all models
- ARIMAModel: ARIMA time series forecasting
- ExponentialSmoothingModel: Coming soon

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

from .base import BaseModel
from .arima import ARIMAModel

__all__ = [
    "BaseModel",
    "ARIMAModel",
]

