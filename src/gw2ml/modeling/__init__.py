"""
GW2ML Modeling Module.

This module provides the forecasting model framework:
- BaseModel: Abstract base class for all models
- Concrete models: ARIMA, ExponentialSmoothing (coming soon)

Example:
    >>> from gw2ml.modeling import BaseModel
    >>> from gw2ml.modeling import ARIMAModel  # coming soon
"""

from .base import BaseModel

__all__ = [
    "BaseModel",
]

