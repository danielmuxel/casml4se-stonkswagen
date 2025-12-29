"""
GW2ML Modeling Module.

This module provides the forecasting model framework:
- BaseModel: Abstract base class for all models
- ARIMAModel: ARIMA time series forecasting
- ExponentialSmoothingModel: Exponential Smoothing (Holt-Winters)
- XGBoostModel: XGBoost machine learning model

Example:
    >>> from gw2ml.modeling import ARIMAModel, XGBoostModel
    >>> from gw2ml.data import load_gw2_series
    >>>
    >>> data = load_gw2_series(19976, days_back=30, value_column="sell_unit_price")
    >>> train, test = data.split(train=0.8)
    >>>
    >>> # ARIMA (klassisch)
    >>> model = ARIMAModel(p=2, d=1, q=2)
    >>> model.fit(train.series)
    >>> forecast = model.predict(n=len(test.series))
    >>>
    >>> # XGBoost (Machine Learning - empfohlen!)
    >>> model = XGBoostModel(lags=12, n_estimators=100)
    >>> model.fit(train.series)
    >>> forecast = model.predict(n=len(test.series))
"""

from .base import BaseModel
from .arima import ARIMAModel
from .exponential_smoothing import ExponentialSmoothingModel
from .xgboost_model import XGBoostModel

__all__ = [
    "BaseModel",
    "ARIMAModel",
    "ExponentialSmoothingModel",
    "XGBoostModel",
]

