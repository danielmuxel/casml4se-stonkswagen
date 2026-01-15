"""
Tests for the gw2ml.modeling.arima module.

These tests cover:
- ARIMAModel instantiation and configuration
- Fit and predict functionality
- SARIMA (seasonal ARIMA)
- Integration with real GW2 data
- Performance metrics calculation
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest
from darts import TimeSeries
from darts.metrics import mape, rmse, mae, smape

from gw2ml.modeling import ARIMAModel, BaseModel


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _create_mock_series(days: int = 30) -> TimeSeries:
    """Create a mock TimeSeries for testing."""
    points_per_day = 288  # 5-minute intervals
    total_points = days * points_per_day

    end_time = datetime.now().replace(microsecond=0, second=0, minute=0)
    start_time = end_time - timedelta(days=days)

    dates = pd.date_range(start=start_time, periods=total_points, freq="5min")
    # Create a pattern with trend and some seasonality
    values = [100 + i * 0.01 + (i % 288) * 0.05 for i in range(total_points)]

    df = pd.DataFrame({"value": values}, index=dates)
    return TimeSeries.from_dataframe(df, value_cols=["value"])


def _resolve_test_item_id() -> int:
    """Get test item ID from environment or use default."""
    env_value = os.getenv("TEST_ITEM_ID")
    if env_value and env_value.isdigit():
        return int(env_value)
    return 19976  # Mystic Coin


# ══════════════════════════════════════════════════════════════════════════════
# TEST: INSTANTIATION
# ══════════════════════════════════════════════════════════════════════════════


class TestARIMAModelInstantiation:
    """Tests for ARIMAModel creation and configuration."""

    def test_inherits_from_base_model(self) -> None:
        """ARIMAModel inherits from BaseModel."""
        model = ARIMAModel()
        assert isinstance(model, BaseModel)

    def test_default_parameters(self) -> None:
        """Default parameters are set correctly."""
        model = ARIMAModel()
        assert model.params["p"] == 1
        assert model.params["d"] == 1
        assert model.params["q"] == 1
        assert model.params["seasonal_order"] is None

    def test_custom_parameters(self) -> None:
        """Custom parameters are stored correctly."""
        model = ARIMAModel(p=2, d=1, q=2)
        assert model.params["p"] == 2
        assert model.params["d"] == 1
        assert model.params["q"] == 2

    def test_seasonal_parameters(self) -> None:
        """Seasonal (SARIMA) parameters are stored correctly."""
        model = ARIMAModel(p=1, d=1, q=1, seasonal_order=(1, 1, 1, 288))
        assert model.params["seasonal_order"] == (1, 1, 1, 288)

    def test_name_property(self) -> None:
        """Name property returns 'ARIMA'."""
        model = ARIMAModel()
        assert model.name == "ARIMA"

    def test_default_params_property(self) -> None:
        """default_params property returns expected defaults."""
        model = ARIMAModel()
        expected = {"p": 1, "d": 1, "q": 1, "seasonal_order": None}
        assert model.default_params == expected


# ══════════════════════════════════════════════════════════════════════════════
# TEST: REPR
# ══════════════════════════════════════════════════════════════════════════════


class TestARIMAModelRepr:
    """Tests for string representation."""

    def test_repr_simple_arima(self) -> None:
        """repr shows ARIMA(p,d,q) notation."""
        model = ARIMAModel(p=2, d=1, q=2)
        assert repr(model) == "ARIMA(2,1,2)"

    def test_repr_sarima(self) -> None:
        """repr shows SARIMA notation with seasonal component."""
        model = ARIMAModel(p=1, d=1, q=1, seasonal_order=(1, 0, 1, 288))
        assert repr(model) == "ARIMA(1,1,1)(1,0,1)[288]"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: FIT
# ══════════════════════════════════════════════════════════════════════════════


class TestARIMAModelFit:
    """Tests for the fit method."""

    def test_fit_creates_model(self) -> None:
        """fit() creates the internal Darts model."""
        series = _create_mock_series(days=10)
        model = ARIMAModel(p=1, d=1, q=1)

        assert model._model is None
        model.fit(series)
        assert model._model is not None

    def test_fit_returns_self(self) -> None:
        """fit() returns self for method chaining."""
        series = _create_mock_series(days=10)
        model = ARIMAModel()

        result = model.fit(series)
        assert result is model

    def test_fit_different_orders(self) -> None:
        """fit() works with different ARIMA orders."""
        series = _create_mock_series(days=10)

        # Note: d=0 can cause issues with some statsmodels versions
        # Using common valid ARIMA configurations with d>=1
        orders = [(1, 1, 0), (1, 1, 1), (2, 1, 2)]
        for p, d, q in orders:
            model = ARIMAModel(p=p, d=d, q=q)
            model.fit(series)
            assert model._model is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PREDICT
# ══════════════════════════════════════════════════════════════════════════════


class TestARIMAModelPredict:
    """Tests for the predict method."""

    def test_predict_returns_timeseries(self) -> None:
        """predict() returns a TimeSeries."""
        series = _create_mock_series(days=10)
        model = ARIMAModel(p=1, d=1, q=1)
        model.fit(series)

        forecast = model.predict(n=12)
        assert isinstance(forecast, TimeSeries)

    def test_predict_correct_length(self) -> None:
        """predict() returns the correct number of steps."""
        series = _create_mock_series(days=10)
        model = ARIMAModel(p=1, d=1, q=1)
        model.fit(series)

        for n in [1, 12, 48, 100]:
            forecast = model.predict(n=n)
            assert len(forecast) == n

    def test_predict_without_fit_raises(self) -> None:
        """predict() raises error if not fitted."""
        model = ARIMAModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(n=12)

    def test_method_chaining(self) -> None:
        """fit() and predict() can be chained."""
        series = _create_mock_series(days=10)
        model = ARIMAModel(p=1, d=1, q=1)

        forecast = model.fit(series).predict(n=12)
        assert len(forecast) == 12


# ══════════════════════════════════════════════════════════════════════════════
# TEST: HISTORICAL FORECASTS
# ══════════════════════════════════════════════════════════════════════════════


class TestARIMAModelHistoricalForecasts:
    """Tests for backtesting with historical_forecasts."""

    def test_historical_forecasts_works(self) -> None:
        """historical_forecasts() produces forecasts."""
        series = _create_mock_series(days=10)
        model = ARIMAModel(p=1, d=1, q=1)
        model.fit(series)

        # Note: ARIMA requires retrain=True (LocalForecastingModel)
        forecasts = model.historical_forecasts(
            series=series,
            start=0.8,
            forecast_horizon=6,
            stride=6,
            retrain=True,
        )

        assert len(forecasts) > 0
        assert isinstance(forecasts, TimeSeries)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: METRICS CALCULATION
# ══════════════════════════════════════════════════════════════════════════════


class TestARIMAModelMetrics:
    """Tests for calculating performance metrics."""

    def test_can_calculate_mape(self) -> None:
        """MAPE can be calculated from forecast."""
        series = _create_mock_series(days=10)
        train, test = series.split_after(0.8)

        model = ARIMAModel(p=1, d=1, q=1)
        model.fit(train)
        forecast = model.predict(n=len(test))

        # Calculate MAPE
        error = mape(test, forecast)
        assert error >= 0  # MAPE is always positive
        assert error < 100  # Should be reasonable for this simple series

    def test_can_calculate_multiple_metrics(self) -> None:
        """Multiple metrics can be calculated."""
        series = _create_mock_series(days=10)
        train, test = series.split_after(0.8)

        model = ARIMAModel(p=1, d=1, q=1)
        model.fit(train)
        forecast = model.predict(n=len(test))

        metrics = {
            "mape": mape(test, forecast),
            "rmse": rmse(test, forecast),
            "mae": mae(test, forecast),
            "smape": smape(test, forecast),
        }

        # All metrics should be calculable
        for name, value in metrics.items():
            assert value >= 0, f"{name} should be non-negative"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: INTEGRATION WITH REAL DATA
# ══════════════════════════════════════════════════════════════════════════════


class TestARIMAModelIntegration:
    """Integration tests with real GW2 data."""

    def test_with_real_gw2_data(self) -> None:
        """Test ARIMA with real GW2 price data."""
        from gw2ml.data import load_gw2_series

        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        # Load data
        data = load_gw2_series(item_id, days_back=days_back)

        # Split
        train, test = data.split(train=0.8)

        # Train model
        model = ARIMAModel(p=1, d=1, q=1)
        model.fit(train)

        # Predict
        forecast = model.predict(n=len(test))

        # Verify
        assert len(forecast) == len(test)
        assert isinstance(forecast, TimeSeries)

    def test_metrics_with_real_data(self) -> None:
        """Calculate metrics with real GW2 data."""
        from gw2ml.data import load_gw2_series

        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        # Load and split
        data = load_gw2_series(item_id, days_back=days_back)
        train, test = data.split(train=0.8)

        # Train and predict
        model = ARIMAModel(p=1, d=1, q=1)
        model.fit(train)
        forecast = model.predict(n=len(test))

        # Calculate metrics
        metrics = {
            "mape": mape(test, forecast),
            "rmse": rmse(test, forecast),
            "mae": mae(test, forecast),
            "smape": smape(test, forecast),
        }

        print(f"\n📊 ARIMA(1,1,1) Performance on item {item_id}:")
        print(f"   MAPE:  {metrics['mape']:.2f}%")
        print(f"   RMSE:  {metrics['rmse']:.2f}")
        print(f"   MAE:   {metrics['mae']:.2f}")
        print(f"   SMAPE: {metrics['smape']:.2f}%")

        # Basic sanity checks
        assert metrics["mape"] < 50, "MAPE should be reasonable"

    def test_compare_different_orders(self) -> None:
        """Compare different ARIMA orders on real data."""
        from gw2ml.data import load_gw2_series

        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        # Load and split
        data = load_gw2_series(item_id, days_back=days_back)
        train, test = data.split(train=0.8)

        # Try different orders (d>=1 for stability)
        orders = [(1, 1, 0), (1, 1, 1), (2, 1, 2)]
        results = []

        for p, d, q in orders:
            model = ARIMAModel(p=p, d=d, q=q)
            model.fit(train)
            forecast = model.predict(n=len(test))
            error = mape(test, forecast)
            results.append((f"ARIMA({p},{d},{q})", error))

        print(f"\n📊 ARIMA Order Comparison on item {item_id}:")
        for name, error in sorted(results, key=lambda x: x[1]):
            print(f"   {name}: MAPE = {error:.2f}%")

        # All should produce valid forecasts
        assert all(error < 100 for _, error in results)

