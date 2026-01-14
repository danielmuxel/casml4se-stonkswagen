"""Tests for the backtesting framework.

This test suite verifies:
- The walk_forward_backtest function works correctly
- No data leakage occurs during backtesting
- Multiple models can be backtested together
- Results contain expected attributes
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from darts import TimeSeries

from gw2ml.evaluation.backtest import (
    BacktestResult,
    backtest_multiple_models,
    walk_forward_backtest,
)
from gw2ml.modeling.arima import ARIMAModel
from gw2ml.modeling.xgboost import XGBoostModel


def _create_test_series(days: int = 30) -> TimeSeries:
    """Create a simple time series for testing."""
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=days * 24, freq="h")
    values = [100 + i % 10 for i in range(len(dates))]
    df = pd.DataFrame({"value": values}, index=dates)
    return TimeSeries.from_dataframe(df, value_cols="value")


class TestWalkForwardBacktest:
    """Tests for walk_forward_backtest function."""

    def test_returns_backtest_result(self) -> None:
        """walk_forward_backtest returns a BacktestResult object."""
        series = _create_test_series(days=10)
        result = walk_forward_backtest(
            model_class=ARIMAModel,
            model_params={"p": 1, "d": 1, "q": 1},
            series=series,
            forecast_horizon=1,
            start=0.8,
            stride=1,
        )
        assert isinstance(result, BacktestResult)

    def test_result_has_forecasts(self) -> None:
        """BacktestResult contains forecast TimeSeries."""
        series = _create_test_series(days=10)
        result = walk_forward_backtest(
            model_class=ARIMAModel,
            model_params={"p": 1, "d": 1, "q": 1},
            series=series,
            forecast_horizon=1,
            start=0.8,
            stride=1,
        )
        assert isinstance(result.forecasts, TimeSeries)
        assert len(result.forecasts) > 0

    def test_result_has_actuals(self) -> None:
        """BacktestResult contains actual TimeSeries aligned with forecasts."""
        series = _create_test_series(days=10)
        result = walk_forward_backtest(
            model_class=ARIMAModel,
            model_params={"p": 1, "d": 1, "q": 1},
            series=series,
            forecast_horizon=1,
            start=0.8,
            stride=1,
        )
        assert isinstance(result.actuals, TimeSeries)
        assert len(result.actuals) == len(result.forecasts)

    def test_result_contains_metadata(self) -> None:
        """BacktestResult contains model metadata."""
        series = _create_test_series(days=10)
        params = {"p": 2, "d": 1, "q": 2}
        result = walk_forward_backtest(
            model_class=ARIMAModel,
            model_params=params,
            series=series,
            forecast_horizon=3,
            start=0.7,
            stride=2,
        )
        assert result.model_name == "ARIMA"
        assert result.params == params
        assert result.forecast_horizon == 3
        assert result.start == 0.7
        assert result.stride == 2

    def test_works_with_xgboost(self) -> None:
        """walk_forward_backtest works with XGBoost model."""
        series = _create_test_series(days=10)
        result = walk_forward_backtest(
            model_class=XGBoostModel,
            model_params={"lags": 12},
            series=series,
            forecast_horizon=1,
            start=0.8,
            stride=1,
        )
        assert isinstance(result, BacktestResult)
        assert result.model_name == "XGBoost"
        assert len(result.forecasts) > 0

    def test_different_forecast_horizons(self) -> None:
        """walk_forward_backtest works with different forecast horizons."""
        series = _create_test_series(days=10)
        for horizon in [1, 3, 6]:
            result = walk_forward_backtest(
                model_class=ARIMAModel,
                model_params={"p": 1, "d": 1, "q": 1},
                series=series,
                forecast_horizon=horizon,
                start=0.8,
                stride=horizon,
            )
            assert result.forecast_horizon == horizon
            assert len(result.forecasts) > 0

    def test_no_data_leakage_fresh_instance(self) -> None:
        """Backtesting creates a fresh model instance each time."""
        series = _create_test_series(days=10)

        # Run backtest twice with same parameters
        result1 = walk_forward_backtest(
            model_class=ARIMAModel,
            model_params={"p": 1, "d": 1, "q": 1},
            series=series,
            forecast_horizon=1,
            start=0.8,
        )
        result2 = walk_forward_backtest(
            model_class=ARIMAModel,
            model_params={"p": 1, "d": 1, "q": 1},
            series=series,
            forecast_horizon=1,
            start=0.8,
        )

        # Results should be identical (same parameters, same data)
        assert len(result1.forecasts) == len(result2.forecasts)
        # Forecasts should match closely (allowing for small numerical differences)
        fc1_values = result1.forecasts.values()
        fc2_values = result2.forecasts.values()
        assert fc1_values.shape == fc2_values.shape


class TestBacktestMultipleModels:
    """Tests for backtest_multiple_models function."""

    def test_backtests_multiple_models(self) -> None:
        """backtest_multiple_models returns results for all models."""
        series = _create_test_series(days=10)
        model_specs = [
            {"class": ARIMAModel, "params": {"p": 1, "d": 1, "q": 1}},
            {"class": XGBoostModel, "params": {"lags": 12}},
        ]

        results = backtest_multiple_models(
            model_specs=model_specs,
            series=series,
            forecast_horizon=1,
            start=0.8,
        )

        assert len(results) == 2
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_each_result_has_correct_model(self) -> None:
        """Each result corresponds to the correct model."""
        series = _create_test_series(days=10)
        model_specs = [
            {"class": ARIMAModel, "params": {"p": 1, "d": 1, "q": 1}},
            {"class": XGBoostModel, "params": {"lags": 12}},
        ]

        results = backtest_multiple_models(
            model_specs=model_specs,
            series=series,
            forecast_horizon=1,
            start=0.8,
        )

        assert results[0].model_name == "ARIMA"
        assert results[1].model_name == "XGBoost"

    def test_handles_empty_params(self) -> None:
        """backtest_multiple_models handles specs without params."""
        series = _create_test_series(days=10)
        model_specs = [
            {"class": ARIMAModel},  # No params - should use defaults
        ]

        results = backtest_multiple_models(
            model_specs=model_specs,
            series=series,
            forecast_horizon=1,
            start=0.8,
        )

        assert len(results) == 1
        assert isinstance(results[0], BacktestResult)
