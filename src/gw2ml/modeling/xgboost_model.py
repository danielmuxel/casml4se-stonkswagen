"""
XGBoost Model for Time Series Forecasting.

XGBoost (eXtreme Gradient Boosting) is a powerful machine learning algorithm
that uses gradient boosting on decision trees. Unlike ARIMA/ES, it can:
- Learn complex non-linear patterns
- Use multiple lag features
- Handle volatile/random data better

Example:
    >>> from gw2ml.modeling import XGBoostModel
    >>> from gw2ml.data import load_gw2_series
    >>>
    >>> data = load_gw2_series(19976, days_back=30, value_column="sell_unit_price")
    >>> train, test = data.split(train=0.8)
    >>>
    >>> model = XGBoostModel(lags=12, n_estimators=100)
    >>> model.fit(train.series)
    >>> forecast = model.predict(n=len(test.series))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from darts import TimeSeries
from darts.models import XGBModel

from .base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost Model for time series forecasting.

    This model wraps Darts' XGBModel and provides a consistent interface
    through BaseModel. Unlike ARIMA/ES, XGBoost:
    - Can learn complex non-linear patterns
    - Uses lag features (past values as predictors)
    - Is a GlobalForecastingModel (can be trained once, predict multiple times)

    Parameters:
        lags: Number of lagged values to use as features.
            E.g., lags=12 means using the past 12 values to predict.
        output_chunk_length: Number of steps to predict at once.
        n_estimators: Number of boosting rounds (trees).
        max_depth: Maximum depth of each tree.
        learning_rate: Step size shrinkage (eta).
        subsample: Fraction of samples used per tree.
        colsample_bytree: Fraction of features used per tree.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.

    Attributes:
        name: "XGBoost"
        default_params: Reasonable defaults for time series forecasting.

    Example:
        >>> # Simple model with 12 lags
        >>> model = XGBoostModel(lags=12)
        >>>
        >>> # More complex model
        >>> model = XGBoostModel(
        ...     lags=24,
        ...     n_estimators=200,
        ...     max_depth=6,
        ...     learning_rate=0.1
        ... )
        >>>
        >>> # Train and predict
        >>> model.fit(train_series)
        >>> forecast = model.predict(n=100)

    Note:
        XGBoost is a GlobalForecastingModel in Darts, which means:
        - It can use retrain=False for historical_forecasts() (faster!)
        - It learns patterns that generalize across the series
        - It's more robust to volatile/random data
    """

    @property
    def name(self) -> str:
        """Model name for logging and identification."""
        return "XGBoost"

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default XGBoost parameters for time series."""
        return {
            "lags": 12,
            "output_chunk_length": 1,
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }

    def __init__(
        self,
        lags: Union[int, List[int]] = 12,
        output_chunk_length: int = 1,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize XGBoost model.

        Args:
            lags: Number of past values to use as features.
                Can be int (uses 1 to lags) or list of specific lags.
            output_chunk_length: Steps to predict at once.
            n_estimators: Number of boosting rounds.
            max_depth: Max tree depth.
            learning_rate: Learning rate (eta).
            subsample: Row sampling ratio.
            colsample_bytree: Column sampling ratio.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            **kwargs: Additional arguments for Darts XGBModel.
        """
        self.params: Dict[str, Any] = {
            "lags": lags,
            "output_chunk_length": output_chunk_length,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            **kwargs,
        }
        self._model: Optional[XGBModel] = None

    def build_model(self, **kwargs: Any) -> XGBModel:
        """
        Build the Darts XGBModel with current parameters.

        Args:
            **kwargs: Override parameters for this build.

        Returns:
            Configured Darts XGBModel.
        """
        # Merge stored params with any overrides
        build_params = {**self.params, **kwargs}

        return XGBModel(**build_params)

    def fit(self, series: TimeSeries, **kwargs: Any) -> "XGBoostModel":
        """
        Fit the XGBoost model on the training series.

        Args:
            series: Training TimeSeries.
            **kwargs: Additional arguments for fit.

        Returns:
            self (for method chaining).

        Example:
            >>> model = XGBoostModel(lags=12)
            >>> model.fit(train_series)
            >>> forecast = model.predict(n=12)
        """
        self._model = self.build_model()
        self._model.fit(series, **kwargs)
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

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: float = 0.5,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: bool = False,  # XGBoost can use False!
        **kwargs: Any,
    ) -> TimeSeries:
        """
        Generate historical forecasts (backtesting).

        Unlike ARIMA/ES, XGBoost can use retrain=False for much faster
        backtesting since it's a GlobalForecastingModel.

        Args:
            series: The TimeSeries to backtest on.
            start: Start point (float = ratio, int = index).
            forecast_horizon: Number of steps to forecast each time.
            stride: Steps between forecasts.
            retrain: Whether to retrain at each step (False is faster!).
            **kwargs: Additional arguments.

        Returns:
            TimeSeries with historical forecasts.
        """
        if self._model is None:
            raise ValueError(
                "Model must be fitted before historical_forecasts. Call fit() first."
            )
        return self._model.historical_forecasts(
            series=series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation with key parameters."""
        lags = self.params.get("lags", 12)
        n_est = self.params.get("n_estimators", 100)
        depth = self.params.get("max_depth", 5)
        lr = self.params.get("learning_rate", 0.1)
        return f"XGBoost(lags={lags}, n_est={n_est}, depth={depth}, lr={lr})"


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = ["XGBoostModel"]

