"""Matplotlib plotting helpers for price and quantity time series."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["plot_prices", "plot_quantity"]


def plot_prices(df: pd.DataFrame) -> plt.Figure:
    """Plot sell and buy prices over time."""
    time_col = "fetched_at"
    sell_price_col = "sell_unit_price"
    buy_price_col = "buy_unit_price"
    fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
    ax.plot(df[time_col], df[sell_price_col], color="tab:red", linewidth=1.8, label="Sell Price")
    ax.plot(df[time_col], df[buy_price_col], color="tab:green", linewidth=1.8, label="Buy Price")
    ax.set_title("Prices over time")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    return fig


def plot_quantity(df: pd.DataFrame, col: str, title: str, color: str) -> plt.Figure:
    """Plot quantity over time for a specific column."""
    time_col = "fetched_at"
    fig, ax = plt.subplots(figsize=(10, 3.0), layout="constrained")
    ax.plot(df[time_col], df[col], color=color, linewidth=1.6, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel("Quantity")
    ax.grid(True, linestyle=":", alpha=0.5)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    return fig

