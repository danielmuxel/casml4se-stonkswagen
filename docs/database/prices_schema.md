# prices Schema

| column_name | data_type | is_nullable | column_default |
| --- | --- | --- | --- |
| id | bigint | NO | nextval('prices_id_seq'::regclass) |
| item_id | integer | NO |  |
| whitelisted | boolean | NO |  |
| buy_quantity | integer | NO |  |
| buy_unit_price | integer | NO |  |
| sell_quantity | integer | NO |  |
| sell_unit_price | integer | NO |  |
| fetched_at | timestamp with time zone | YES | CURRENT_TIMESTAMP |
| created_at | timestamp with time zone | YES | CURRENT_TIMESTAMP |

## Usage

```python
from datetime import UTC, datetime

from gw2ml.data import DatabaseClient, get_prices


client = DatabaseClient.from_env()

# Fetch the last six hours of price data for a single item.
recent_prices = get_prices(client, item_id=19702, last_hours=6)

# Pull an explicit date range (UTC-aware datetimes recommended).
weekly_prices = get_prices(
    client,
    item_id=19702,
    start_time=datetime(2025, 1, 1, tzinfo=UTC),
    end_time=datetime(2025, 1, 8, tzinfo=UTC),
    limit=500,
)
```
