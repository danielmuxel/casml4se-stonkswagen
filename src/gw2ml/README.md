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

