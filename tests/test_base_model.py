"""
Tests for the gw2ml.modeling.base module.

These tests cover:
- BaseModel cannot be instantiated directly (ABC)
- Concrete implementations work correctly
- All interface methods function as expected
- Method chaining works
- Backtesting via historical_forecasts
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import pytest
from darts import TimeSeries
from darts.models import NaiveSeasonal

from gw2ml.modeling import BaseModel


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
    # Create a simple pattern: daily cycle with some noise
    values = [100 + (i % 288) * 0.1 for i in range(total_points)]

    df = pd.DataFrame({"value": values}, index=dates)
    return TimeSeries.from_dataframe(df, value_cols=["value"])


# ══════════════════════════════════════════════════════════════════════════════
# DUMMY MODEL: Concrete implementation for testing
# ══════════════════════════════════════════════════════════════════════════════


class DummyModel(BaseModel):
    """
    Minimal concrete implementation of BaseModel for testing.

    Uses NaiveSeasonal internally - the simplest Darts model.
    This just repeats the last K values as the forecast.
    """

    @property
    def name(self) -> str:
        return "Dummy"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"K": 1}

    def __init__(self, K: int = 1) -> None:
        """
        Args:
            K: Seasonal period for NaiveSeasonal (default: 1 = repeat last value)
        """
        self.params = {"K": K}
        self._model: NaiveSeasonal | None = None

    def build_model(self, **kwargs: Any) -> NaiveSeasonal:
        """Build a NaiveSeasonal model with the configured K."""
        return NaiveSeasonal(K=self.params["K"])

    def fit(self, series: TimeSeries, **kwargs: Any) -> "DummyModel":
        """Fit the model on the series."""
        self._model = self.build_model()
        self._model.fit(series)
        return self

    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        """Generate forecast for n steps."""
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(n)


class DummyModelWithCovariates(DummyModel):
    """Dummy model that claims to support covariates."""

    @property
    def supports_covariates(self) -> bool:
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ABC BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseModelABC:
    """Tests for ABC (Abstract Base Class) behavior."""

    def test_cannot_instantiate_base_model(self) -> None:
        """BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseModel()  # type: ignore

    def test_incomplete_implementation_fails(self) -> None:
        """Incomplete implementations cannot be instantiated."""

        class IncompleteModel(BaseModel):
            @property
            def name(self) -> str:
                return "Incomplete"

            # Missing: default_params, build_model, fit, predict

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel()  # type: ignore

    def test_complete_implementation_works(self) -> None:
        """Complete implementations can be instantiated."""
        model = DummyModel()
        assert model is not None
        assert isinstance(model, BaseModel)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseModelProperties:
    """Tests for model properties."""

    def test_name_property(self) -> None:
        """Name property returns correct value."""
        model = DummyModel()
        assert model.name == "Dummy"

    def test_default_params_property(self) -> None:
        """Default params property returns correct dict."""
        model = DummyModel()
        assert model.default_params == {"K": 1}

    def test_supports_covariates_default_false(self) -> None:
        """supports_covariates defaults to False."""
        model = DummyModel()
        assert model.supports_covariates is False

    def test_supports_covariates_can_be_overridden(self) -> None:
        """supports_covariates can be overridden in subclass."""
        model = DummyModelWithCovariates()
        assert model.supports_covariates is True


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PARAMETER METHODS
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseModelParams:
    """Tests for parameter getter/setter methods."""

    def test_get_params_returns_current_params(self) -> None:
        """get_params returns the current parameters."""
        model = DummyModel(K=5)
        assert model.get_params() == {"K": 5}

    def test_get_params_with_default(self) -> None:
        """get_params works with default parameters."""
        model = DummyModel()
        assert model.get_params() == {"K": 1}

    def test_set_params_updates_params(self) -> None:
        """set_params updates parameters."""
        model = DummyModel(K=1)
        model.set_params(K=10)
        assert model.get_params() == {"K": 10}

    def test_set_params_returns_self(self) -> None:
        """set_params returns self for method chaining."""
        model = DummyModel()
        result = model.set_params(K=5)
        assert result is model

    def test_set_params_adds_new_params(self) -> None:
        """set_params can add new parameters."""
        model = DummyModel()
        model.set_params(new_param="test")
        assert model.get_params() == {"K": 1, "new_param": "test"}


# ══════════════════════════════════════════════════════════════════════════════
# TEST: FIT METHOD
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseModelFit:
    """Tests for the fit method."""

    def test_fit_trains_model(self) -> None:
        """fit() trains the internal model."""
        series = _create_mock_series(days=10)
        model = DummyModel(K=288)

        assert model._model is None
        model.fit(series)
        assert model._model is not None

    def test_fit_returns_self(self) -> None:
        """fit() returns self for method chaining."""
        series = _create_mock_series(days=10)
        model = DummyModel()

        result = model.fit(series)
        assert result is model

    def test_fit_method_chaining(self) -> None:
        """fit() can be chained with predict()."""
        series = _create_mock_series(days=10)
        model = DummyModel(K=288)

        # Should work in one line
        forecast = model.fit(series).predict(n=12)
        assert len(forecast) == 12


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PREDICT METHOD
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseModelPredict:
    """Tests for the predict method."""

    def test_predict_returns_timeseries(self) -> None:
        """predict() returns a TimeSeries."""
        series = _create_mock_series(days=10)
        model = DummyModel(K=288)
        model.fit(series)

        forecast = model.predict(n=12)
        assert isinstance(forecast, TimeSeries)

    def test_predict_correct_length(self) -> None:
        """predict() returns correct number of steps."""
        series = _create_mock_series(days=10)
        model = DummyModel(K=288)
        model.fit(series)

        for n in [1, 12, 100]:
            forecast = model.predict(n=n)
            assert len(forecast) == n

    def test_predict_without_fit_raises(self) -> None:
        """predict() raises error if model not fitted."""
        model = DummyModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(n=12)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: HISTORICAL FORECASTS (BACKTESTING)
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseModelHistoricalForecasts:
    """Tests for the historical_forecasts method (backtesting).

    Note: NaiveSeasonal is a LocalForecastingModel and requires retrain=True
    for historical_forecasts. GlobalForecastingModels (like XGBoost, LSTM)
    support retrain=False.
    """

    def test_historical_forecasts_returns_timeseries(self) -> None:
        """historical_forecasts() returns a TimeSeries."""
        series = _create_mock_series(days=10)
        model = DummyModel(K=288)
        model.fit(series)

        # LocalForecastingModels require retrain=True
        forecasts = model.historical_forecasts(
            series=series,
            start=0.8,
            forecast_horizon=6,
            stride=6,
            retrain=True,  # Required for NaiveSeasonal
        )

        assert isinstance(forecasts, TimeSeries)

    def test_historical_forecasts_produces_forecasts(self) -> None:
        """historical_forecasts() produces non-empty forecasts."""
        series = _create_mock_series(days=10)
        model = DummyModel(K=288)
        model.fit(series)

        forecasts = model.historical_forecasts(
            series=series,
            start=0.8,
            forecast_horizon=6,
            stride=6,
            retrain=True,  # Required for NaiveSeasonal
        )

        assert len(forecasts) > 0

    def test_historical_forecasts_with_different_horizons(self) -> None:
        """historical_forecasts() works with different forecast horizons."""
        series = _create_mock_series(days=10)
        model = DummyModel(K=288)
        model.fit(series)

        for horizon in [3, 6, 12]:
            forecasts = model.historical_forecasts(
                series=series,
                start=0.8,
                forecast_horizon=horizon,
                stride=horizon,
                retrain=True,
            )
            assert len(forecasts) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: REPR
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseModelRepr:
    """Tests for string representation."""

    def test_repr_contains_name(self) -> None:
        """__repr__ contains the model name."""
        model = DummyModel()
        assert "Dummy" in repr(model)

    def test_repr_contains_params(self) -> None:
        """__repr__ contains the parameters."""
        model = DummyModel(K=5)
        repr_str = repr(model)
        assert "K=5" in repr_str

    def test_repr_format(self) -> None:
        """__repr__ has expected format."""
        model = DummyModel(K=3)
        assert repr(model) == "Dummy(K=3)"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseModelIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self) -> None:
        """Test complete train → predict workflow."""
        # Create data
        series = _create_mock_series(days=30)

        # Create model
        model = DummyModel(K=288)
        assert model.name == "Dummy"
        assert model.get_params() == {"K": 288}

        # Train
        model.fit(series)

        # Predict
        forecast = model.predict(n=24)  # 2 hours ahead
        assert len(forecast) == 24

    def test_full_workflow_with_backtest(self) -> None:
        """Test complete workflow including backtesting."""
        series = _create_mock_series(days=10)

        model = DummyModel(K=288)
        model.fit(series)

        # Backtest on last 20% of data
        # Note: retrain=True required for LocalForecastingModels
        forecasts = model.historical_forecasts(
            series=series,
            start=0.8,
            forecast_horizon=6,
            stride=6,
            retrain=True,
        )

        assert len(forecasts) > 0

    def test_multiple_models_same_interface(self) -> None:
        """Multiple model implementations share the same interface."""
        series = _create_mock_series(days=10)

        models = [
            DummyModel(K=1),
            DummyModel(K=288),
            DummyModelWithCovariates(K=288),
        ]

        for model in models:
            # All should have same interface
            assert hasattr(model, "name")
            assert hasattr(model, "default_params")
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

            # All should work
            model.fit(series)
            forecast = model.predict(n=6)
            assert len(forecast) == 6

