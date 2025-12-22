# src/gw2ml

Shared preprocessing steps, feature engineering helpers, logging utilities, and configuration primitives belong here. Anything reused across multiple environments or model families should live in this package.

## Database helpers

The `database` submodule exposes a `DatabaseClient` wrapper plus query helpers that accept either the client instance or a raw connection string.

```python
from datetime import datetime, UTC, timedelta

from gw2ml import DatabaseClient, database


client = DatabaseClient.from_env()

# Prices for a specific item during the last day.
last_day = datetime.now(UTC) - timedelta(days=1)
prices_df = database.get_prices(client, item_id=19721, start_time=last_day)

# Fetch a custom table range; this works for historical tables as well.
history_df = database.get_generic_rows(
    client,
    table="gw2tp_historical_prices",
    item_id=19721,
    start_time=last_day,
    time_column="timestamp",
    time_column_unit="milliseconds",
)
```

Instantiate the client once when your application starts and pass it around instead of repeatedly opening new engines.

## Pipeline building blocks

- `pipeline/data_preparation.py` exposes `PipelineContext`, `load_training_snapshot`, and `load_latest_validation`. Pipelines request deterministic training snapshots (point-in-time) together with a rolling validation slice that always targets the freshest data.
- `pipeline/feature_hooks.py` keeps optional augmentors organized per phase (`training`, `validation`, `inference`). Register a callable with `register_augmentor("training", hook)` and the pipeline will apply it automatically when assembling datasets.
- `pipeline/train_arima.py` shows the expected orchestration flow: build a context, download training + validation data, run optional augmentors, fit a model, evaluate, and persist artifacts (datasets + serialized model).

## Serving surfaces

Serving endpoints live under `gw2ml/serving/<model>/`. Each module mounts a FastAPI router and mirrors the feature hooks from training (using the `inference` phase). The ARIMA example demonstrates how to:

1. Load the latest approved artifact via `gw2ml.modeling.load_model`.
2. Fetch fresh validation rows on-demand if the caller requests `fetch_latest`.
3. Apply inference augmentors before generating predictions.
4. Return both predictions and lightweight metadata (item id, timestamp).

Promote any future model by copying this pattern and wiring its router into the hosting application.

