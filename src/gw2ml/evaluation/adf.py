"""Augmented Dickey-Fuller (ADF) test for stationarity."""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def perform_adf_test(series: pd.Series) -> dict:
    """
    Perform the Augmented Dickey-Fuller test on a pandas Series.

    Returns a dictionary with the results.
    """
    # Remove NaN values as they can break the test
    series = series.dropna()

    if len(series) < 20:
        return {
            "error": "Series too short for reliable ADF test (min 20 samples required)."
        }

    # Check for constant series (no variance)
    if series.nunique(dropna=True) <= 1 or series.std() == 0:
        return {
            "error": "Series is constant (no variance). ADF test requires varying data."
        }

    try:
        result = adfuller(series)
    except ValueError as exc:
        return {
            "error": f"ADF test failed: {exc}"
        }
    
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "icbest": result[5] if len(result) > 5 else None,
        "is_stationary": result[1] < 0.05
    }
