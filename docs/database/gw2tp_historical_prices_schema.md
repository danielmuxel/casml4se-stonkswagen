# gw2tp_historical_prices Schema

| column_name | data_type | is_nullable | column_default |
| --- | --- | --- | --- |
| id | bigint | NO | nextval('gw2tp_historical_prices_id_seq'::regclass) |
| item_id | integer | NO |  |
| timestamp | bigint | NO |  |
| sell_price | integer | YES |  |
| buy_price | integer | YES |  |
| supply | integer | YES |  |
| demand | integer | YES |  |
| created_at | timestamp with time zone | YES | CURRENT_TIMESTAMP |

## Usage

```python
from ml.common.database import DatabaseClient, get_tp_history


client = DatabaseClient.from_env()

# Grab the most recent 12 hours of TP history.
latest_tp = get_tp_history(client, item_id=19702, last_hours=12)

# Limit by date range if you need a precise comparison window.
from datetime import UTC, datetime

tp_slice = get_tp_history(
    client,
    item_id=19702,
    start_time=datetime(2025, 2, 1, tzinfo=UTC),
    end_time=datetime(2025, 2, 7, tzinfo=UTC),
    limit=5000,
)
```
