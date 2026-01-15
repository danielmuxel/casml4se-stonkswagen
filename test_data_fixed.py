#!/usr/bin/env python
"""Test script to check data quality."""

import pandas as pd
from gw2ml.data.loaders import load_gw2_series

series_meta = load_gw2_series(
    item_id=19976,
    days_back=30,
    value_column="sell_unit_price",
    fill_missing_dates=True,
)

print(f"Total points: {series_meta.num_points}")
print(f"Start: {series_meta.start_time}")
print(f"End: {series_meta.end_time}")

# Convert to pandas for inspection
df = series_meta.series.pd_dataframe()
print(f"\nDataFrame shape: {df.shape}")
print(f"NaN count: {df.isna().sum().sum()}")
print(f"Inf count: {(df == float('inf')).sum().sum() + (df == float('-inf')).sum().sum()}")

# Check for validation split
train_ratio = 0.7
val_ratio = 0.15

n = len(series_meta.series)
train_len = int(n * train_ratio)
val_len = int(n * val_ratio)
test_len = n - train_len - val_len

print(f"\nSplit sizes:")
print(f"  Train: {train_len}")
print(f"  Val: {val_len}")
print(f"  Test: {test_len}")

# Check train split
parts = series_meta.split(train=train_ratio, val=val_ratio)
train_ts, val_ts, test_ts = parts

print(f"\nActual split lengths:")
print(f"  Train: {len(train_ts)}")
print(f"  Val: {len(val_ts)}")
print(f"  Test: {len(test_ts)}")

# Check for NaNs in each split
train_df = train_ts.pd_dataframe()
val_df = val_ts.pd_dataframe()
test_df = test_ts.pd_dataframe()

print(f"\nNaNs in splits:")
print(f"  Train: {train_df.isna().sum().sum()}")
print(f"  Val: {val_df.isna().sum().sum()}")
print(f"  Test: {test_df.isna().sum().sum()}")

print(f"\nFirst few values of train:")
print(train_df.head(20))
