"""
Tests for the gw2ml.data.loaders module.

These tests cover:
- GW2Series split methods (ratio-based and day-based)
- Loading functions with real database
- Caching behavior
- Edge cases and error handling
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest
from darts import TimeSeries

from gw2ml.data import (
    GW2Series,
    clear_cache,
    get_cache_info,
    load_and_split,
    load_and_split_days,
    load_gw2_series,
    load_gw2_series_batch,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_test_item_id() -> int:
    """Get test item ID from environment or use default."""
    env_value = os.getenv("TEST_ITEM_ID")
    if env_value and env_value.isdigit():
        return int(env_value)
    return 19697  # Copper Ore


def _create_mock_series(days: int = 30) -> TimeSeries:
    """Create a mock TimeSeries for testing split logic."""
    # 288 points per day (5-minute intervals)
    points_per_day = 288
    total_points = days * points_per_day

    end_time = datetime.now().replace(microsecond=0, second=0)
    start_time = end_time - timedelta(days=days)

    # Use freq="5min" to create properly spaced time series
    dates = pd.date_range(start=start_time, periods=total_points, freq="5min")
    values = list(range(total_points))  # Simple incrementing values

    df = pd.DataFrame({"value": values}, index=dates)
    return TimeSeries.from_dataframe(df, value_cols=["value"])


def _create_mock_gw2series(days: int = 30) -> GW2Series:
    """Create a mock GW2Series for testing."""
    series = _create_mock_series(days)

    # Calculate times based on actual series data
    points_per_day = 288
    total_points = days * points_per_day
    start_time = datetime.now().replace(microsecond=0, second=0) - timedelta(days=days)
    end_time = start_time + timedelta(minutes=5 * (total_points - 1))

    return GW2Series(
        series=series,
        item_id=99999,
        item_name="Test Item",
        value_column="buy_unit_price",
        start_time=start_time,
        end_time=end_time,
        num_points=len(series),
    )


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: GW2Series.split() - Ratio-based
# ══════════════════════════════════════════════════════════════════════════════


class TestGW2SeriesSplitRatio:
    """Tests for ratio-based split method."""

    def test_split_2way_default(self) -> None:
        """Test 2-way split with default 80/20 ratio."""
        data = _create_mock_gw2series(days=30)
        train, test = data.split()

        total = len(data.series)
        expected_train = int(total * 0.8)

        assert len(train) == expected_train
        assert len(test) == total - expected_train
        assert len(train) + len(test) == total

    def test_split_2way_custom_ratio(self) -> None:
        """Test 2-way split with custom ratio."""
        data = _create_mock_gw2series(days=30)
        train, test = data.split(train=0.7)

        total = len(data.series)
        expected_train = int(total * 0.7)

        assert len(train) == expected_train
        assert len(test) == total - expected_train

    def test_split_3way(self) -> None:
        """Test 3-way split with train/val/test."""
        data = _create_mock_gw2series(days=30)
        train, val, test = data.split(train=0.7, val=0.15)

        total = len(data.series)

        # Check that all parts together equal the original
        assert len(train) + len(val) + len(test) == total

        # Check approximate ratios (may not be exact due to rounding)
        assert len(train) == int(total * 0.7)

    def test_split_3way_ratios_sum_to_one(self) -> None:
        """Test 3-way split where train + val + test = 1.0."""
        data = _create_mock_gw2series(days=30)
        train, val, test = data.split(train=0.6, val=0.2)
        # test should be 0.2

        total = len(data.series)
        assert len(train) + len(val) + len(test) == total

    def test_split_invalid_ratios(self) -> None:
        """Test that invalid ratios raise ValueError."""
        data = _create_mock_gw2series(days=30)

        with pytest.raises(ValueError, match="must be less than 1.0"):
            data.split(train=0.8, val=0.3)  # Sum = 1.1


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: GW2Series.split_days() - Day-based
# ══════════════════════════════════════════════════════════════════════════════


class TestGW2SeriesSplitDays:
    """Tests for day-based split method."""

    def test_split_days_2way(self) -> None:
        """Test 2-way split by days."""
        data = _create_mock_gw2series(days=30)
        train, test = data.split_days(test_days=7)

        # Test should be approximately 7 days worth of data
        # At 288 points/day, that's about 2016 points
        points_per_day = 288
        expected_test_points = 7 * points_per_day

        # Allow some tolerance due to time boundaries
        assert abs(len(test) - expected_test_points) < points_per_day
        assert len(train) + len(test) == len(data.series)

    def test_split_days_3way(self) -> None:
        """Test 3-way split by days."""
        data = _create_mock_gw2series(days=30)
        train, val, test = data.split_days(test_days=7, val_days=3)

        # All parts should sum to total
        assert len(train) + len(val) + len(test) == len(data.series)

        # Check approximate sizes
        points_per_day = 288
        assert abs(len(test) - 7 * points_per_day) < points_per_day
        assert abs(len(val) - 3 * points_per_day) < points_per_day

    def test_split_days_too_many_days(self) -> None:
        """Test that requesting too many days raises ValueError."""
        data = _create_mock_gw2series(days=30)

        with pytest.raises(ValueError, match="must be less than total_days"):
            data.split_days(test_days=25, val_days=10)  # 35 > 30

    def test_split_days_edge_case_all_test(self) -> None:
        """Test with most data in test set."""
        data = _create_mock_gw2series(days=30)
        train, test = data.split_days(test_days=25)

        # Should have small train set
        assert len(train) > 0
        assert len(test) > len(train)


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: GW2Series properties and methods
# ══════════════════════════════════════════════════════════════════════════════


class TestGW2SeriesProperties:
    """Tests for GW2Series properties and utility methods."""

    def test_total_days(self) -> None:
        """Test total_days property."""
        data = _create_mock_gw2series(days=30)
        assert abs(data.total_days - 30) < 1  # Allow 1 day tolerance

    def test_info_string(self) -> None:
        """Test info() method returns readable string."""
        data = _create_mock_gw2series(days=30)
        info = data.info()

        assert "Test Item" in info
        assert "99999" in info
        assert "buy_unit_price" in info
        assert "Data points:" in info

    def test_repr(self) -> None:
        """Test __repr__ method."""
        data = _create_mock_gw2series(days=30)
        repr_str = repr(data)

        assert "GW2Series" in repr_str
        assert "item_id=99999" in repr_str


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS: Real database loading
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadGW2Series:
    """Integration tests for load_gw2_series with real database."""

    @pytest.fixture(autouse=True)
    def clear_cache_before_test(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_load_single_item(self) -> None:
        """Test loading a single item."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        data = load_gw2_series(item_id, days_back=days_back)

        assert data.item_id == item_id
        assert data.num_points > 0
        assert isinstance(data.series, TimeSeries)
        assert data.value_column == "buy_unit_price"

    def test_load_with_caching(self) -> None:
        """Test that caching works correctly."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        # First load
        data1 = load_gw2_series(item_id, days_back=days_back)
        cache_info = get_cache_info()
        assert cache_info["entries"] == 1

        # Second load should hit cache
        data2 = load_gw2_series(item_id, days_back=days_back)

        # Should be the same object (from cache)
        assert data1 is data2

    def test_load_without_caching(self) -> None:
        """Test loading without cache."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        data1 = load_gw2_series(item_id, days_back=days_back, use_cache=False)
        data2 = load_gw2_series(item_id, days_back=days_back, use_cache=False)

        # Should be different objects
        assert data1 is not data2
        # But same data
        assert data1.num_points == data2.num_points

    def test_load_sell_price(self) -> None:
        """Test loading sell price instead of buy price."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        data = load_gw2_series(
            item_id, days_back=days_back, value_column="sell_unit_price"
        )

        assert data.value_column == "sell_unit_price"

    def test_load_invalid_item(self) -> None:
        """Test loading non-existent item raises ValueError."""
        with pytest.raises(ValueError, match="No data found"):
            load_gw2_series(item_id=999999999, days_back=7)


class TestLoadGW2SeriesBatch:
    """Integration tests for batch loading."""

    @pytest.fixture(autouse=True)
    def clear_cache_before_test(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_load_batch(self) -> None:
        """Test loading multiple items."""
        # Common tradeable items
        item_ids = [19697, 19699, 19700]  # Copper, Iron, Mithril ore
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        batch = load_gw2_series_batch(item_ids, days_back=days_back)

        # Should have data for at least some items
        assert len(batch) > 0

        for item_id, data in batch.items():
            assert item_id in item_ids
            assert isinstance(data, GW2Series)
            assert data.num_points > 0

    def test_load_batch_uses_cache(self) -> None:
        """Test that batch loading populates cache."""
        item_ids = [19697, 19699]
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        load_gw2_series_batch(item_ids, days_back=days_back)

        cache_info = get_cache_info()
        assert cache_info["entries"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS: Convenience functions
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadAndSplit:
    """Integration tests for convenience functions."""

    @pytest.fixture(autouse=True)
    def clear_cache_before_test(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_load_and_split_2way(self) -> None:
        """Test load_and_split with 2-way split."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        train, test = load_and_split(item_id, days_back=days_back, train=0.8)

        assert isinstance(train, TimeSeries)
        assert isinstance(test, TimeSeries)
        assert len(train) > len(test)

    def test_load_and_split_3way(self) -> None:
        """Test load_and_split with 3-way split."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        train, val, test = load_and_split(
            item_id, days_back=days_back, train=0.7, val=0.15
        )

        assert isinstance(train, TimeSeries)
        assert isinstance(val, TimeSeries)
        assert isinstance(test, TimeSeries)

    def test_load_and_split_days_2way(self) -> None:
        """Test load_and_split_days with 2-way split."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "14"))

        train, test = load_and_split_days(
            item_id, days_back=days_back, test_days=3
        )

        assert isinstance(train, TimeSeries)
        assert isinstance(test, TimeSeries)

    def test_load_and_split_days_3way(self) -> None:
        """Test load_and_split_days with 3-way split."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "14"))

        train, val, test = load_and_split_days(
            item_id, days_back=days_back, test_days=3, val_days=2
        )

        assert isinstance(train, TimeSeries)
        assert isinstance(val, TimeSeries)
        assert isinstance(test, TimeSeries)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestCache:
    """Tests for cache functionality."""

    def test_clear_cache(self) -> None:
        """Test that clear_cache empties the cache."""
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))

        # Load something to populate cache
        load_gw2_series(item_id, days_back=days_back)
        assert get_cache_info()["entries"] > 0

        # Clear
        clear_cache()
        assert get_cache_info()["entries"] == 0

    def test_cache_info(self) -> None:
        """Test get_cache_info returns correct structure."""
        clear_cache()

        info = get_cache_info()
        assert "entries" in info
        assert "item_ids" in info
        assert info["entries"] == 0

        # Load something
        item_id = _resolve_test_item_id()
        days_back = int(os.getenv("TEST_DAYS_BACK", "7"))
        load_gw2_series(item_id, days_back=days_back)

        info = get_cache_info()
        assert info["entries"] == 1
        assert info["item_ids"] == 1

