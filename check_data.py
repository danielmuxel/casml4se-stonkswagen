#!/usr/bin/env python
from gw2ml.data.loaders import load_gw2_series

series_meta = load_gw2_series(19697, days_back=30, value_column="sell_unit_price", fill_missing_dates=True)
print(f"Points: {series_meta.num_points}")

df = series_meta.series.pd_dataframe()
print(f"Shape: {df.shape}")
print(f"NaNs: {df.isna().sum().sum()}")
print(f"\nFirst 10 rows:")
print(df.head(10))
