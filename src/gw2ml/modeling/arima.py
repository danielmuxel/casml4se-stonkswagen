from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class MedianPriceModel:
    """
    Lightweight baseline that memorizes per-item median prices.

    This is intentionally simple so we always have a deterministic artifact to
    persist from pipelines even when a more sophisticated ARIMA configuration
    is not yet wired up.
    """

    value_column: str = "sell_unit_price"
    fallback_value: float = 0.0
    medians_by_item: dict[int, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "MedianPriceModel":
        """Compute medians per item from the supplied dataframe."""
        if df.empty:
            message = "Cannot fit MedianPriceModel on an empty dataframe."
            raise ValueError(message)
        required_columns = {"item_id", self.value_column}
        missing = required_columns.difference(df.columns)
        if missing:
            message = f"Missing required columns: {', '.join(sorted(missing))}"
            raise ValueError(message)

        grouped = (
            df.dropna(subset=["item_id", self.value_column])
            .groupby("item_id")[self.value_column]
            .median()
        )
        medians = {int(item_id): float(value) for item_id, value in grouped.items()}
        if not medians:
            message = "No medians computed; check input filters."
            raise ValueError(message)

        self.medians_by_item = medians
        self.fallback_value = float(grouped.median()) if not grouped.empty else 0.0
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return per-row predictions aligned with the incoming dataframe."""
        if not self.medians_by_item:
            message = "MedianPriceModel has not been fitted yet."
            raise ValueError(message)
        if "item_id" not in df.columns:
            message = "Prediction dataframe must include an item_id column."
            raise ValueError(message)

        predictions = df["item_id"].map(self.medians_by_item)
        if predictions.isna().any():
            predictions = predictions.fillna(self.fallback_value)
        return predictions.astype(float)

    def to_payload(self) -> dict[str, Any]:
        """Serialize fitted state to a JSON-friendly payload."""
        return {
            "value_column": self.value_column,
            "fallback_value": self.fallback_value,
            "medians_by_item": self.medians_by_item,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MedianPriceModel":
        """Rehydrate a model from :meth:`to_payload` output."""
        model = cls(
            value_column=payload.get("value_column", "sell_unit_price"),
            fallback_value=float(payload.get("fallback_value", 0.0)),
        )
        raw_medians = payload.get("medians_by_item", {})
        model.medians_by_item = {int(item_id): float(value) for item_id, value in raw_medians.items()}
        return model


__all__ = ["MedianPriceModel"]



