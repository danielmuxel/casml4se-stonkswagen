"""
Tests for the gw2ml.modeling.xgboost_model module.

These tests cover:
- XGBoostModel instantiation and configuration
- Fit and predict functionality
- Different lag configurations
- Integration with real GW2 data (sell_unit_price)
- Performance metrics calculation
- Comparison with classical models
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest
from darts import TimeSeries
import numpy as np

from gw2ml.modeling import XGBoostModel, BaseModel


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
    values = [100 + i * 0.01 + (i % 288) * 0.05 + np.random.randn() * 0.5 
              for i in range(total_points)]

    df = pd.DataFrame({"value": values}, index=dates)
    return TimeSeries.from_dataframe(df, value_cols=["value"])


def _resolve_test_item_id() -> int:
    """Get test item ID from environment or use default."""
    env_value = os.getenv("TEST_ITEM_ID")
    if env_value and env_value.isdigit():
        return int(env_value)
    return 19976  # Mystic Coin (volatile!)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: INSTANTIATION
# ══════════════════════════════════════════════════════════════════════════════


class TestXGBoostModelInstantiation:
    """Tests for XGBoostModel creation and configuration."""

    def test_inherits_from_base_model(self) -> None:
        """XGBoostModel inherits from BaseModel."""
        model = XGBoostModel()
        assert isinstance(model, BaseModel)

    def test_default_parameters(self) -> None:
        """Default parameters are set correctly."""
        model = XGBoostModel()
        assert model.params["lags"] == 12
        assert model.params["output_chunk_length"] == 1
        assert model.params["n_estimators"] == 100
        assert model.params["max_depth"] == 5
        assert model.params["learning_rate"] == 0.1

    def test_custom_parameters(self) -> None:
        """Custom parameters are stored correctly."""
        model = XGBoostModel(
            lags=24,
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05
        )
        assert model.params["lags"] == 24
        assert model.params["n_estimators"] == 200
        assert model.params["max_depth"] == 8
        assert model.params["learning_rate"] == 0.05

    def test_list_lags(self) -> None:
        """Lags can be specified as a list."""
        model = XGBoostModel(lags=[1, 6, 12, 24])
        assert model.params["lags"] == [1, 6, 12, 24]

    def test_name_property(self) -> None:
        """Name property returns 'XGBoost'."""
        model = XGBoostModel()
        assert model.name == "XGBoost"

    def test_default_params_property(self) -> None:
        """default_params property returns expected defaults."""
        model = XGBoostModel()
        defaults = model.default_params
        assert defaults["lags"] == 12
        assert defaults["n_estimators"] == 100


# ══════════════════════════════════════════════════════════════════════════════
# TEST: REPR
# ══════════════════════════════════════════════════════════════════════════════


class TestXGBoostModelRepr:
    """Tests for string representation."""

    def test_repr_default(self) -> None:
        """repr shows key parameters."""
        model = XGBoostModel()
        assert "XGBoost" in repr(model)
        assert "lags=12" in repr(model)
        assert "n_est=100" in repr(model)

    def test_repr_custom(self) -> None:
        """repr reflects custom parameters."""
        model = XGBoostModel(lags=24, n_estimators=200, max_depth=8)
        assert "lags=24" in repr(model)
        assert "n_est=200" in repr(model)
        assert "depth=8" in repr(model)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: FIT
# ══════════════════════════════════════════════════════════════════════════════


class TestXGBoostModelFit:
    """Tests for the fit method."""

    def test_fit_creates_model(self) -> None:
        """fit() creates the internal Darts model."""
        series = _create_mock_series(days=10)
        model = XGBoostModel(lags=6)

        assert model._model is None
        model.fit(series)
        assert model._model is not None

    def test_fit_returns_self(self) -> None:
        """fit() returns self for method chaining."""
        series = _create_mock_series(days=10)
        model = XGBoostModel(lags=6)

        result = model.fit(series)
        assert result is model

    def test_fit_different_lags(self) -> None:
        """fit() works with different lag values."""
        series = _create_mock_series(days=10)

        for lags in [6, 12, 24]:
            model = XGBoostModel(lags=lags)
            model.fit(series)
            assert model._model is not None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PREDICT
# ══════════════════════════════════════════════════════════════════════════════


class TestXGBoostModelPredict:
    """Tests for the predict method."""

    def test_predict_returns_timeseries(self) -> None:
        """predict() returns a TimeSeries."""
        series = _create_mock_series(days=10)
        model = XGBoostModel(lags=6)
        model.fit(series)

        forecast = model.predict(n=12)
        assert isinstance(forecast, TimeSeries)

    def test_predict_correct_length(self) -> None:
        """predict() returns the correct number of steps."""
        series = _create_mock_series(days=10)
        model = XGBoostModel(lags=6)
        model.fit(series)

        for n in [1, 12, 48, 100]:
            forecast = model.predict(n=n)
            assert len(forecast) == n

    def test_predict_without_fit_raises(self) -> None:
        """predict() raises error if not fitted."""
        model = XGBoostModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(n=12)

    def test_method_chaining(self) -> None:
        """fit() and predict() can be chained."""
        series = _create_mock_series(days=10)
        model = XGBoostModel(lags=6)

        forecast = model.fit(series).predict(n=12)
        assert len(forecast) == 12

    def test_predict_not_all_nan(self) -> None:
        """Forecast should have actual values, not NaN."""
        series = _create_mock_series(days=10)
        model = XGBoostModel(lags=6)
        model.fit(series)

        forecast = model.predict(n=50)
        values = forecast.values().flatten()

        # XGBoost should produce finite values
        assert np.all(np.isfinite(values)), "XGBoost should produce finite forecasts"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: HISTORICAL FORECASTS
# ══════════════════════════════════════════════════════════════════════════════


class TestXGBoostModelHistoricalForecasts:
    """Tests for backtesting with historical_forecasts."""

    def test_historical_forecasts_works(self) -> None:
        """historical_forecasts() produces forecasts."""
        series = _create_mock_series(days=10)
        model = XGBoostModel(lags=6)
        model.fit(series)

        # XGBoost can use retrain=False (much faster!)
        forecasts = model.historical_forecasts(
            series=series,
            start=0.8,
            forecast_horizon=6,
            stride=6,
            retrain=False,  # This is the advantage of XGBoost!
        )

        assert len(forecasts) > 0
        assert isinstance(forecasts, TimeSeries)

    def test_historical_forecasts_with_retrain(self) -> None:
        """historical_forecasts() works with retrain=True too."""
        series = _create_mock_series(days=5)  # Smaller for speed
        model = XGBoostModel(lags=6)
        model.fit(series)

        forecasts = model.historical_forecasts(
            series=series,
            start=0.8,
            forecast_horizon=3,
            stride=3,
            retrain=True,
        )

        assert len(forecasts) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: METRICS CALCULATION
# ══════════════════════════════════════════════════════════════════════════════


class TestXGBoostModelMetrics:
    """Tests for calculating performance metrics."""

    def test_can_calculate_metrics(self) -> None:
        """Metrics can be calculated from forecast."""
        series = _create_mock_series(days=10)
        train, test = series.split_after(0.8)

        model = XGBoostModel(lags=12)
        model.fit(train)
        forecast = model.predict(n=len(test))

        # Calculate metrics manually
        test_vals = test.values().flatten()
        fc_vals = forecast.values().flatten()
        min_len = min(len(test_vals), len(fc_vals))

        mae = np.mean(np.abs(test_vals[:min_len] - fc_vals[:min_len]))
        rmse = np.sqrt(np.mean((test_vals[:min_len] - fc_vals[:min_len]) ** 2))

        assert mae >= 0
        assert rmse >= 0
        assert np.isfinite(mae), "MAE should be finite"
        assert np.isfinite(rmse), "RMSE should be finite"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: INTEGRATION WITH REAL DATA
# ══════════════════════════════════════════════════════════════════════════════


class TestXGBoostModelIntegration:
    """Integration tests with real GW2 data."""

    def test_with_real_gw2_data(self) -> None:
        """Test XGBoost with real GW2 price data (sell_unit_price)."""
        from gw2ml.data import load_gw2_series
        from darts import TimeSeries as DartsTimeSeries

        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        # Load data with sell_unit_price
        data = load_gw2_series(
            item_id, 
            days_back=days_back,
            value_column="sell_unit_price"
        )

        # Remove NaN values by converting to DataFrame and dropping
        df = data.series.to_dataframe()
        df = df.dropna()
        if len(df) < 100:
            pytest.skip("Not enough data after removing NaN")
        
        # Reset index to have continuous 5min frequency
        df = df.reset_index(drop=True)
        df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="5min")
        clean_series = DartsTimeSeries.from_dataframe(df)

        # Split
        train, test = clean_series.split_after(0.8)

        # Train model
        model = XGBoostModel(lags=12)
        model.fit(train)

        # Predict
        forecast = model.predict(n=len(test))

        # Verify
        assert len(forecast) == len(test)
        assert isinstance(forecast, TimeSeries)

        # Check forecast is not NaN
        fc_vals = forecast.values().flatten()
        assert np.all(np.isfinite(fc_vals)), "XGBoost should produce finite forecasts"

    def test_metrics_with_real_data(self) -> None:
        """Calculate metrics with real GW2 data."""
        from gw2ml.data import load_gw2_series
        from darts import TimeSeries as DartsTimeSeries

        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        # Load and clean data (sell_unit_price)
        data = load_gw2_series(
            item_id, 
            days_back=days_back,
            value_column="sell_unit_price"
        )
        
        # Remove NaN values
        df = data.series.to_dataframe()
        df = df.dropna()
        if len(df) < 100:
            pytest.skip("Not enough data after removing NaN")
        
        # Reset index to have continuous 5min frequency
        df = df.reset_index(drop=True)
        df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="5min")
        clean_series = DartsTimeSeries.from_dataframe(df)
        train, test = clean_series.split_after(0.8)

        # Train and predict
        model = XGBoostModel(lags=12, n_estimators=50)  # Fewer trees for speed
        model.fit(train)
        forecast = model.predict(n=len(test))

        # Calculate metrics
        test_vals = test.values().flatten()
        fc_vals = forecast.values().flatten()
        min_len = min(len(test_vals), len(fc_vals))

        mape = np.mean(np.abs((test_vals[:min_len] - fc_vals[:min_len]) / test_vals[:min_len])) * 100
        rmse = np.sqrt(np.mean((test_vals[:min_len] - fc_vals[:min_len]) ** 2))
        mae = np.mean(np.abs(test_vals[:min_len] - fc_vals[:min_len]))

        print(f"\n📊 {model} Performance on item {item_id} (sell_unit_price):")
        print(f"   MAPE:  {mape:.2f}%")
        print(f"   RMSE:  {rmse:.2f}")
        print(f"   MAE:   {mae:.2f}")

        # XGBoost should produce valid metrics
        assert np.isfinite(mape), "MAPE should be finite"
        assert np.isfinite(rmse), "RMSE should be finite"

    def test_compare_different_lags(self) -> None:
        """Compare different lag configurations on real data."""
        from gw2ml.data import load_gw2_series
        from darts import TimeSeries as DartsTimeSeries

        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        # Load and clean data
        data = load_gw2_series(
            item_id, 
            days_back=days_back,
            value_column="sell_unit_price"
        )
        
        # Remove NaN values
        df = data.series.to_dataframe()
        df = df.dropna()
        if len(df) < 100:
            pytest.skip("Not enough data after removing NaN")
        
        # Reset index to have continuous 5min frequency
        df = df.reset_index(drop=True)
        df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="5min")
        clean_series = DartsTimeSeries.from_dataframe(df)
        train, test = clean_series.split_after(0.8)

        # Try different lags
        lag_configs = [6, 12, 24]
        results = []

        for lags in lag_configs:
            model = XGBoostModel(lags=lags, n_estimators=50)
            model.fit(train)
            forecast = model.predict(n=len(test))
            
            test_vals = test.values().flatten()
            fc_vals = forecast.values().flatten()
            min_len = min(len(test_vals), len(fc_vals))
            
            mape = np.mean(np.abs((test_vals[:min_len] - fc_vals[:min_len]) / test_vals[:min_len])) * 100
            results.append((f"lags={lags}", mape))

        print(f"\n📊 XGBoost Lag Comparison on item {item_id}:")
        for name, error in sorted(results, key=lambda x: x[1]):
            print(f"   {name}: MAPE = {error:.2f}%")

        # All should produce finite metrics
        assert all(np.isfinite(error) for _, error in results)

