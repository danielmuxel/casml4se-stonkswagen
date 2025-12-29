#!/usr/bin/env python
"""Test script to debug why only ARIMA is being trained."""

from gw2ml.pipelines.train import train_items
from gw2ml.modeling.registry import list_models

print("Available models:", list_models())

# Test with a specific configuration
config = {
    "data": {
        "days_back": 30,
        "value_column": "sell_unit_price",
    },
    "models": [
        {"name": "ARIMA", "grid": None},
        {"name": "ExponentialSmoothing", "grid": None},
        {"name": "XGBoost", "grid": None},
    ],
}

print("\nTraining with config:")
print(config)

results = train_items([19697], override_config=config)

print("\nTraining results:")
for result in results:
    print(f"Item {result['item_id']}: {result['model_name']}")
    if 'models' in result:
        print(f"  Trained models: {[m['model_name'] for m in result['models']]}")
