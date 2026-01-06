"""
High-level data loading utilities for GW2 time series.

This module provides convenient functions to load GW2 price data
and convert it to Darts TimeSeries for forecasting.

Features:
- Simple one-line loading: load_gw2_series(item_id)
- Batch loading: load_gw2_series_batch([item_id1, item_id2])
- Flexible splits: ratio-based or day-based
- Automatic caching for repeated queries
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from darts import TimeSeries

from .database_client import DatabaseClient
from .database_queries import get_items, get_prices


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

_cache: dict[tuple[int, int, str, bool], "GW2Series"] = {}


def clear_cache() -> None:
    """Clear the data cache."""
    _cache.clear()


def get_cache_info() -> dict[str, int]:
    """Get cache statistics."""
    return {
        "entries": len(_cache),
        "item_ids": len(set(key[0] for key in _cache.keys())),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GW2Series DATACLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GW2Series:
    """
    Wrapper for a GW2 TimeSeries with metadata and split methods.

    Attributes:
        series: The Darts TimeSeries object
        item_id: GW2 Item ID
        item_name: Human-readable item name (if available)
        value_column: Which price column was used
        start_time: First timestamp in the series
        end_time: Last timestamp in the series
        num_points: Number of data points

    Example:
        >>> data = load_gw2_series(19697, days_back=30)
        >>> print(data.info())
        >>> train, test = data.split(train=0.8)
        >>> train, val, test = data.split(train=0.7, val=0.15)
    """

    series: TimeSeries
    item_id: int
    item_name: Optional[str]
    value_column: str
    start_time: datetime
    end_time: datetime
    num_points: int

    @property
    def total_days(self) -> float:
        """Total time span in days."""
        return (self.end_time - self.start_time).total_seconds() / 86400

    # ══════════════════════════════════════════
    # RATIO-BASED SPLIT
    # ══════════════════════════════════════════

    def split(
        self,
        train: float = 0.8,
        val: Optional[float] = None,
    ) -> tuple[TimeSeries, ...]:
        """
        Split the series by ratio.

        Args:
            train: Ratio for training set (e.g., 0.8 = 80%)
            val: Ratio for validation set (None = no validation, 2-way split)

        Returns:
            (train, test) if val is None
            (train, val, test) if val is specified

        Raises:
            ValueError: If ratios don't sum to less than 1.0

        Example:
            >>> train, test = data.split(train=0.8)
            >>> train, val, test = data.split(train=0.7, val=0.15)
        """
        if val is None:
            # 2-way split
            train_series, test_series = self.series.split_after(train)
            return train_series, test_series

        # 3-way split
        if train + val >= 1.0:
            raise ValueError(
                f"train ({train}) + val ({val}) must be less than 1.0, "
                f"got {train + val}"
            )

        train_series, rest = self.series.split_after(train)
        val_ratio_of_rest = val / (1 - train)
        val_series, test_series = rest.split_after(val_ratio_of_rest)

        return train_series, val_series, test_series

    # ══════════════════════════════════════════
    # DAY-BASED SPLIT
    # ══════════════════════════════════════════

    def split_days(
        self,
        test_days: int,
        val_days: Optional[int] = None,
    ) -> tuple[TimeSeries, ...]:
        """
        Split the series by days from the end.

        The last `test_days` become the test set,
        the `val_days` before that become validation,
        and the rest becomes training.

        Args:
            test_days: Days for test set (from the end)
            val_days: Days for validation set (None = no validation)

        Returns:
            (train, test) if val_days is None
            (train, val, test) if val_days is specified

        Raises:
            ValueError: If test_days + val_days >= total_days

        Example:
            >>> # 30 days of data, last 7 days for test
            >>> train, test = data.split_days(test_days=7)
            >>>
            >>> # With validation: last 7 days test, 3 days before for val
            >>> train, val, test = data.split_days(test_days=7, val_days=3)
        """
        total = self.total_days
        val_days_safe = val_days or 0

        if test_days + val_days_safe >= total:
            raise ValueError(
                f"test_days ({test_days}) + val_days ({val_days_safe}) "
                f"must be less than total_days ({total:.1f})"
            )

        # Calculate split points based on time
        test_start = self.end_time - timedelta(days=test_days)

        if val_days is None:
            # 2-way split
            train_series, test_series = self.series.split_before(
                pd.Timestamp(test_start)
            )
            return train_series, test_series

        # 3-way split
        val_start = test_start - timedelta(days=val_days)

        train_series, rest = self.series.split_before(pd.Timestamp(val_start))
        val_series, test_series = rest.split_before(pd.Timestamp(test_start))

        return train_series, val_series, test_series

    # ══════════════════════════════════════════
    # INFO METHODS
    # ══════════════════════════════════════════

    def info(self) -> str:
        """Return a human-readable summary."""
        name_display = self.item_name or f"Item {self.item_id}"
        return (
            f"GW2Series({name_display})\n"
            f"  Item ID: {self.item_id}\n"
            f"  Period: {self.start_time:%Y-%m-%d %H:%M} to {self.end_time:%Y-%m-%d %H:%M}\n"
            f"  Days: {self.total_days:.1f}\n"
            f"  Data points: {self.num_points}\n"
            f"  Value column: {self.value_column}"
        )

    def __repr__(self) -> str:
        return f"GW2Series(item_id={self.item_id}, points={self.num_points})"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def _get_db_client() -> DatabaseClient:
    """Get database client from environment."""
    return DatabaseClient.from_env()


def _df_to_series(
    df: pd.DataFrame,
    time_col: str = "fetched_at",
    value_col: str = "buy_unit_price",
    freq: str = "5min",
    fill_missing_dates: bool = True,
) -> TimeSeries:
    """
    Convert a DataFrame to a Darts TimeSeries.

    Handles timezone conversion and index setup automatically.
    Rounds timestamps to the nearest frequency boundary and handles duplicates.

    Args:
        df: Input DataFrame
        time_col: Name of the time column
        value_col: Name of the value column
        freq: Expected frequency of the data (default: "5min" for GW2 data)
        fill_missing_dates: Whether to interpolate missing dates (default: True)
    """
    df = df.copy()

    # Ensure datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Remove timezone (Darts doesn't like timezone-aware timestamps)
    if df[time_col].dt.tz is not None:
        df[time_col] = df[time_col].dt.tz_localize(None)

    # Round timestamps to the nearest frequency boundary
    # This handles the case where data was collected at slightly different times
    df[time_col] = df[time_col].dt.round(freq)

    # Remove duplicates (keep last value for each timestamp)
    df = df.drop_duplicates(subset=[time_col], keep="last")

    # Set index and sort
    df = df.set_index(time_col).sort_index()

    # Keep only the value column
    df = df[[value_col]]

    # Create TimeSeries with explicit frequency
    # GW2 data is collected every 5 minutes, so we use "5min" as the frequency.
    # NOTE: Even with fill_missing_dates=False, Darts may require a known `freq`
    # when it cannot infer it from an irregular/gappy index.

    # If fill_missing_dates is True, we need to handle NaN values for ML models like XGBoost
    # First pass: create TimeSeries with NaN for missing dates
    kwargs = {
        "df": df,
        "value_cols": [value_col],
        "fill_missing_dates": fill_missing_dates,
        "freq": freq,
    }

    ts = TimeSeries.from_dataframe(**kwargs)

    # If fill_missing_dates is True, fill NaN values using pandas operations
    if fill_missing_dates:
        # Convert to pandas, fill NaNs, convert back to TimeSeries
        ts_df = ts.to_dataframe()
        # Forward fill then backward fill to handle all NaNs
        ts_df = ts_df.ffill().bfill()
        # Recreate TimeSeries from cleaned dataframe
        ts = TimeSeries.from_dataframe(ts_df, value_cols=[value_col], freq=freq)

    return ts


def _get_item_name(client: DatabaseClient, item_id: int) -> Optional[str]:
    """Get item name from database."""
    try:
        items_df = get_items(client, item_ids=[item_id], limit=1)
        if not items_df.empty and "name" in items_df.columns:
            return str(items_df.iloc[0]["name"])
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOADING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def load_gw2_series(
    item_id: int,
    days_back: int = 30,
    value_column: str = "buy_unit_price",
    use_cache: bool = True,
    fill_missing_dates: bool = True,
    resample_freq: Optional[str] = None,
) -> GW2Series:
    """
    Load GW2 price data for a single item.

    Args:
        item_id: GW2 Item ID (e.g., 19697 for Copper Ore)
        days_back: Number of days to load
        value_column: "buy_unit_price" or "sell_unit_price"
        use_cache: Whether to use cached data (default: True)
        fill_missing_dates: Whether to interpolate missing dates (default: True)
        resample_freq: Resample frequency for downsampling (e.g., '1H', '6H', '1D')
                       - '1H': hourly (default for raw data)
                       - '6H': 6-hourly (reduces points by 6x)
                       - '1D': daily (reduces points by 24x if hourly data)
                       - None: no resampling (default)

    Returns:
        GW2Series with TimeSeries and metadata

    Raises:
        ValueError: If no data is found for the item

    Example:
        >>> # Load with daily aggregation (faster training)
        >>> data = load_gw2_series(19697, days_back=30, resample_freq='1D')
        >>> print(data.info())
        >>> train, test = data.split(train=0.8)
    """
    # NOTE: fill_missing_dates and resample_freq materially change the resulting TimeSeries index,
    # so they must be part of the cache key.
    cache_key = (item_id, days_back, value_column, fill_missing_dates, resample_freq)

    # Check cache
    if use_cache and cache_key in _cache:
        return _cache[cache_key]

    # Load from database
    client = _get_db_client()

    df = get_prices(client, item_id, last_days=days_back, order="ASC")

    if df.empty:
        raise ValueError(
            f"No data found for item_id={item_id} in the last {days_back} days"
        )

    # Convert to TimeSeries
    series = _df_to_series(df, value_col=value_column, fill_missing_dates=fill_missing_dates)

    # Resample if requested (for faster training with large datasets)
    if resample_freq is not None:
        # Convert to pandas DataFrame, resample, then back to TimeSeries
        df_series = series.pd_dataframe()
        df_resampled = df_series.resample(resample_freq).mean()  # Average values in each bucket
        series = TimeSeries.from_dataframe(df_resampled)

    # Get item name
    item_name = _get_item_name(client, item_id)

    # Get time bounds from the actual data
    start_time = df["fetched_at"].min()
    end_time = df["fetched_at"].max()

    # Handle timezone for datetime objects
    if hasattr(start_time, "tz_localize") and start_time.tzinfo is not None:
        start_time = start_time.tz_localize(None)
    if hasattr(end_time, "tz_localize") and end_time.tzinfo is not None:
        end_time = end_time.tz_localize(None)

    # Convert to Python datetime if needed
    if isinstance(start_time, pd.Timestamp):
        start_time = start_time.to_pydatetime()
    if isinstance(end_time, pd.Timestamp):
        end_time = end_time.to_pydatetime()

    result = GW2Series(
        series=series,
        item_id=item_id,
        item_name=item_name,
        value_column=value_column,
        start_time=start_time,
        end_time=end_time,
        num_points=len(series),
    )

    # Cache result
    if use_cache:
        _cache[cache_key] = result

    return result


def load_gw2_series_batch(
    item_ids: list[int],
    days_back: int = 30,
    value_column: str = "buy_unit_price",
    use_cache: bool = True,
    fill_missing_dates: bool = True,
) -> dict[int, GW2Series]:
    """
    Load GW2 price data for multiple items.

    Args:
        item_ids: List of GW2 Item IDs
        days_back: Number of days to load (same for all items)
        value_column: "buy_unit_price" or "sell_unit_price"
        use_cache: Whether to use cached data
        fill_missing_dates: Whether to interpolate missing dates (default: True)

    Returns:
        Dict mapping item_id to GW2Series

    Example:
        >>> items = [19697, 19699, 19700]  # Copper, Iron, Mithril
        >>> batch = load_gw2_series_batch(items, days_back=30)
        >>>
        >>> for item_id, data in batch.items():
        ...     print(f"{data.item_name}: {data.num_points} points")
    """
    results: dict[int, GW2Series] = {}

    for item_id in item_ids:
        try:
            results[item_id] = load_gw2_series(
                item_id=item_id,
                days_back=days_back,
                value_column=value_column,
                use_cache=use_cache,
                fill_missing_dates=fill_missing_dates,
            )
        except ValueError as e:
            # Skip items with no data, but print warning
            print(f"Warning: Skipping item {item_id}: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def load_and_split(
    item_id: int,
    days_back: int = 30,
    train: float = 0.8,
    val: Optional[float] = None,
    value_column: str = "buy_unit_price",
    fill_missing_dates: bool = True,
) -> tuple[TimeSeries, ...]:
    """
    Load data and split by ratio in one step.

    Args:
        item_id: GW2 Item ID
        days_back: Number of days to load
        train: Ratio for training set
        val: Ratio for validation set (None = 2-way split)
        value_column: "buy_unit_price" or "sell_unit_price"
        fill_missing_dates: Whether to interpolate missing dates (default: True)

    Returns:
        (train, test) if val is None
        (train, val, test) if val is specified

    Example:
        >>> train, test = load_and_split(19697, train=0.8)
        >>> train, val, test = load_and_split(19697, train=0.7, val=0.15)
    """
    data = load_gw2_series(item_id, days_back, value_column, fill_missing_dates=fill_missing_dates)
    return data.split(train=train, val=val)


def load_and_split_days(
    item_id: int,
    days_back: int = 30,
    test_days: int = 7,
    val_days: Optional[int] = None,
    value_column: str = "buy_unit_price",
    fill_missing_dates: bool = True,
) -> tuple[TimeSeries, ...]:
    """
    Load data and split by days in one step.

    Args:
        item_id: GW2 Item ID
        days_back: Total days to load
        test_days: Days for test set (from the end)
        val_days: Days for validation set (None = 2-way split)
        value_column: "buy_unit_price" or "sell_unit_price"
        fill_missing_dates: Whether to interpolate missing dates (default: True)

    Returns:
        (train, test) if val_days is None
        (train, val, test) if val_days is specified

    Example:
        >>> train, test = load_and_split_days(19697, days_back=30, test_days=7)
        >>> train, val, test = load_and_split_days(
        ...     19697, days_back=30, test_days=7, val_days=3
        ... )
    """
    data = load_gw2_series(item_id, days_back, value_column, fill_missing_dates=fill_missing_dates)
    return data.split_days(test_days=test_days, val_days=val_days)


def list_items(search: str | None = None, limit: int = 100) -> list[dict[str, str]]:
    """
    List items from the database.

    Args:
        search: Optional search string (fuzzy search on name or ID)
        limit: Maximum number of items to return

    Returns:
        List of dictionaries with item_id and item_name
    """
    client = _get_db_client()
    df = get_items(client, search=search, limit=limit)
    
    items = []
    for _, row in df.iterrows():
        items.append({
            "item_id": str(row["id"]),
            "item_name": str(row["name"])
        })
    return items


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "GW2Series",
    "load_gw2_series",
    "load_gw2_series_batch",
    "load_and_split",
    "load_and_split_days",
    "list_items",
    "clear_cache",
    "get_cache_info",
]

