from __future__ import annotations

import pandas as pd
import pytest

from gw2ml.modeling import MedianPriceModel


def test_median_price_model_predicts_per_item() -> None:
    df = pd.DataFrame(
        {
            "item_id": [1, 1, 2, 2, 2],
            "sell_unit_price": [10, 20, 5, 15, 25],
        }
    )
    model = MedianPriceModel().fit(df)
    to_predict = pd.DataFrame({"item_id": [1, 2, 3]})
    predictions = model.predict(to_predict)
    assert predictions.tolist() == [15.0, 15.0, pytest.approx(model.fallback_value)]


def test_median_price_model_requires_value_column() -> None:
    model = MedianPriceModel()
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame({"item_id": [1]}))




