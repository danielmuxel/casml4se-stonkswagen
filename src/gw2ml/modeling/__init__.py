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
from .chronos import Chronos2
from .registry import get_default_grid, get_model, list_models, register_model
from .xgboost import XGBoostModel

# Default grids used when a model-specific grid is not provided via config.
# ARIMA grid reduced to single configuration for faster training
# ARIMA(1,1,1) is a good default for most time series
_ARIMA_DEFAULT_GRID = {
    "p": [1],  # Reduced from [0, 1, 2] - autoregressive order
    "d": [1],  # Reduced from [0, 1] - differencing order
    "q": [1],  # Reduced from [0, 1, 2] - moving average order
    "seasonal_order": [(0, 0, 0, 0)],  # No seasonality
}
# This reduces grid search from 18 combinations to just 1
# For more thorough search, increase to: p=[1,2], d=[1], q=[1,2] (4 combinations)

# ExponentialSmoothing requires special handling - can't grid search independently
# because seasonal and seasonal_periods must be coordinated
_ES_DEFAULT_GRID = {
    "trend": [None, "add"],
    "damped": [False],
    "seasonal": [None],  # Only None by default to avoid requiring seasonal_periods
    "seasonal_periods": [None],  # Match seasonal=None
}

_XGB_DEFAULT_GRID = {
    "lags": [12, 24],
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "output_chunk_length": [1],
    "reg_alpha": [0.0],
    "reg_lambda": [1.0],
}

# Register built-in models on import; new models can register here too.
register_model("ARIMA", ARIMAModel, default_grid=_ARIMA_DEFAULT_GRID)
register_model(
    "ExponentialSmoothing",
    ExponentialSmoothingModel,
    default_grid=_ES_DEFAULT_GRID,
)
register_model("XGBoost", XGBoostModel, default_grid=_XGB_DEFAULT_GRID)
register_model("Chronos2", Chronos2)
# Example for future additions:
# from .nbeats import NBEATSModel
# register_model("NBEATS", NBEATSModel, default_grid={"num_blocks": [2, 3]})

__all__ = [
    "BaseModel",
    "ARIMAModel",
    "ExponentialSmoothingModel",
    "XGBoostModel",
    "Chronos2",
    "register_model",
    "get_model",
    "get_default_grid",
    "list_models",
]

