"""Walk-forward backtesting framework with guaranteed data leakage prevention.

This module provides a centralized backtesting utility that ensures:
1. No data leakage - models only see past data when making predictions
2. Consistent evaluation across all model types
3. Proper walk-forward validation methodology
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union

from darts import TimeSeries

from gw2ml.modeling.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from walk-forward backtesting.

    Attributes:
        forecasts: TimeSeries containing historical forecasts
        actuals: TimeSeries containing actual values aligned with forecasts
        model_name: Name of the model used
        params: Parameters used to create the model
        forecast_horizon: Number of steps ahead predicted at each point
        stride: Number of steps between forecasts
        start: Starting point (ratio or index) for backtesting
        train_size: Number of points used for training (if applicable)
        test_size: Number of points used for testing (if applicable)
    """
    forecasts: TimeSeries
    actuals: TimeSeries
    model_name: str
    params: Dict[str, Any]
    forecast_horizon: int
    stride: int
    start: Union[float, int]
    train_size: int = 0
    test_size: int = 0


def walk_forward_backtest(
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    series: TimeSeries,
    forecast_horizon: int = 1,
    start: Union[float, int] = 0.8,
    stride: int = 1,
    verbose: bool = False,
    train_series: TimeSeries | None = None,
    test_series: TimeSeries | None = None,
) -> BacktestResult:
    """
    Perform walk-forward backtesting with guaranteed data leakage prevention.

    This function creates a FRESH model instance to ensure no contamination from
    any previously fitted models. It then performs proper walk-forward validation
    where at each point in time, the model only uses past data to make predictions.

    Two modes of operation:
    1. **Full series mode** (train_series=None, test_series=None):
       - Uses 'start' parameter to split series for walk-forward validation
       - Backtests from 'start' point to end of series

    2. **Train/Test split mode** (train_series and test_series provided):
       - Trains ONLY on train_series (no data leakage from test)
       - Generates forecasts for the test_series period
       - Proper out-of-sample evaluation

    Args:
        model_class: The model class to test (e.g., ARIMAModel, XGBoostModel)
        model_params: Parameters to pass to the model constructor
        series: The full time series (used if train/test not provided)
        forecast_horizon: Number of steps ahead to forecast at each point (default: 1)
        start: Where to start backtesting, as ratio 0.0-1.0 or absolute index (default: 0.8)
        stride: Number of steps to move forward between forecasts (default: 1)
        verbose: Whether to print progress information (default: False)
        train_series: Optional training data only (for train/test split mode)
        test_series: Optional test data only (for train/test split mode)

    Returns:
        BacktestResult containing forecasts, actuals, and metadata

    Examples:
        >>> # Mode 1: Full series with walk-forward
        >>> result = walk_forward_backtest(
        ...     model_class=ARIMAModel,
        ...     model_params={"p": 1, "d": 1, "q": 1},
        ...     series=my_series,
        ...     forecast_horizon=1,
        ...     start=0.8,
        ... )

        >>> # Mode 2: Train/test split evaluation
        >>> result = walk_forward_backtest(
        ...     model_class=ARIMAModel,
        ...     model_params={"p": 1, "d": 1, "q": 1},
        ...     series=full_series,  # Full series for context
        ...     train_series=train_data,
        ...     test_series=test_data,
        ...     forecast_horizon=1,
        ... )

    Notes:
        - Always creates a fresh model instance to prevent data leakage
        - In train/test mode, model NEVER sees test data during training
        - For 1-step-ahead forecasts (forecast_horizon=1), predictions should closely
          track actual values if the model is good - this is expected behavior
        - For multi-step forecasts (forecast_horizon>1), there will be more visible lag
    """
    # CRITICAL: Create a completely fresh model instance with no fitted state.
    # This ensures zero possibility of data leakage from any previously fitted models.
    fresh_model = model_class(**model_params)

    # Determine which mode to use
    use_train_test_split = train_series is not None and test_series is not None

    if use_train_test_split:
        # MODE 2: Train/Test Split Evaluation
        # Train ONLY on training data, then forecast into test period
        if verbose:
            logger.info(f"Starting train/test split backtest for {model_class.__name__}")
            logger.info(f"  Train size: {len(train_series)}, Test size: {len(test_series)}")
            logger.info(f"  Forecast horizon: {forecast_horizon}, Stride: {stride}")

        # Fit model on training data ONLY
        logger.debug(f"Fitting {fresh_model.name} on training data only...")
        fresh_model.fit(train_series)

        # Generate rolling forecasts for the test period
        # We'll forecast step-by-step through the test period, retraining as we go
        logger.debug(f"Generating rolling forecasts for test period...")

        # Combine train + test for historical_forecasts to work properly
        combined_series = train_series.append(test_series)

        # Start forecasting from the end of train data
        train_end_idx = len(train_series)

        forecasts = fresh_model.historical_forecasts(
            series=combined_series,
            start=train_end_idx,  # Start at end of training data
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=True,  # Retrain at each step using only past data
            last_points_only=False,  # Return all predictions, not just last point of each window
            verbose=verbose,
        )

        # Handle list of TimeSeries (when last_points_only=False with stride >= 1)
        if isinstance(forecasts, list):
            from darts import concatenate
            forecasts = concatenate(forecasts, axis=0)

        # Extract actuals from test period
        actuals = test_series.slice_intersect(forecasts)

        train_size = len(train_series)
        test_size = len(test_series)

    else:
        # MODE 1: Full Series Walk-Forward Validation
        if verbose:
            logger.info(f"Starting walk-forward backtest for {model_class.__name__}")
            logger.info(f"  Params: {model_params}")
            logger.info(f"  Forecast horizon: {forecast_horizon}, Start: {start}, Stride: {stride}")

        # Perform walk-forward backtesting with retrain=True
        # At each step, the model will be fitted using ONLY past data
        logger.debug(f"Running historical_forecasts (retrain=True) for {fresh_model.name}")
        forecasts = fresh_model.historical_forecasts(
            series=series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=True,  # ALWAYS retrain to prevent data leakage
            last_points_only=False,  # Return all predictions, not just last point of each window
            verbose=verbose,
        )

        # Handle list of TimeSeries (when last_points_only=False with stride >= 1)
        if isinstance(forecasts, list):
            from darts import concatenate
            forecasts = concatenate(forecasts, axis=0)

        # Extract the corresponding actual values
        actuals = series.slice_intersect(forecasts)

        train_size = 0
        test_size = 0

    if verbose:
        logger.info(f"Backtest complete: {len(forecasts)} forecast points generated")

    return BacktestResult(
        forecasts=forecasts,
        actuals=actuals,
        model_name=fresh_model.name,
        params=model_params,
        forecast_horizon=forecast_horizon,
        stride=stride,
        start=start,
        train_size=train_size,
        test_size=test_size,
    )


def backtest_multiple_models(
    model_specs: List[Dict[str, Any]],
    series: TimeSeries,
    forecast_horizon: int = 1,
    start: Union[float, int] = 0.8,
    stride: int = 1,
    verbose: bool = False,
) -> List[BacktestResult]:
    """
    Backtest multiple models on the same series.

    Args:
        model_specs: List of dicts with "class" and "params" keys
        series: The full time series to backtest on
        forecast_horizon: Number of steps ahead to forecast (default: 1)
        start: Where to start backtesting (default: 0.8)
        stride: Steps between forecasts (default: 1)
        verbose: Print progress (default: False)

    Returns:
        List of BacktestResult objects, one per model

    Example:
        >>> from gw2ml.modeling.arima import ARIMAModel
        >>> from gw2ml.modeling.xgboost import XGBoostModel
        >>>
        >>> results = backtest_multiple_models(
        ...     model_specs=[
        ...         {"class": ARIMAModel, "params": {"p": 1, "d": 1, "q": 1}},
        ...         {"class": XGBoostModel, "params": {"lags": 12}},
        ...     ],
        ...     series=my_series,
        ... )
    """
    results = []
    for spec in model_specs:
        model_class = spec["class"]
        model_params = spec.get("params", {})

        result = walk_forward_backtest(
            model_class=model_class,
            model_params=model_params,
            series=series,
            forecast_horizon=forecast_horizon,
            start=start,
            stride=stride,
            verbose=verbose,
        )
        results.append(result)

    return results


__all__ = ["walk_forward_backtest", "backtest_multiple_models", "BacktestResult"]
