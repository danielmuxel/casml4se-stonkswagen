#!/usr/bin/env python
"""Test forecasting with all models."""

from gw2ml.pipelines.forecast import forecast_item

# Test forecasting with all trained models
result = forecast_item(
    item_id=19976,
    override_config={
        "data": {"days_back": 30, "value_column": "sell_unit_price"},
        "forecast": {"horizon": 12},
        "models": [
            {"name": "ARIMA", "grid": None},
            {"name": "ExponentialSmoothing", "grid": None},
            {"name": "XGBoost", "grid": None},
        ],
    },
    retrain=False,
)

print("Forecast result:")
print(f"  Item ID: {result['item_id']}")
print(f"  Horizon: {result['forecast_horizon']}")
print(f"  Primary metric: {result['primary_metric']}")
print(f"  Missing models: {result['missing_models']}")
print(f"\nModels forecasted:")
for model in result['models']:
    hist_metric = model.get('history', {}).get('metric', 'N/A')
    future_len = len(model.get('future', {}).get('values', []))
    metric_str = f"{hist_metric:.4f}" if isinstance(hist_metric, float) else str(hist_metric)
    print(f"  - {model['model_name']}: backtest {result['primary_metric']}={metric_str}, future points={future_len}")
