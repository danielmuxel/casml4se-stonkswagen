# gw2bltc_historical_prices Schema

| column_name | data_type | is_nullable | column_default |
| --- | --- | --- | --- |
| id | bigint | NO | nextval('gw2bltc_historical_prices_id_seq'::regclass) |
| item_id | integer | NO |  |
| timestamp | bigint | NO |  |
| sell_price | integer | NO |  |
| buy_price | integer | NO |  |
| supply | integer | NO |  |
| demand | integer | NO |  |
| sold | integer | NO |  |
| offers | integer | NO |  |
| bought | integer | NO |  |
| bids | integer | NO |  |
| created_at | timestamp with time zone | YES | CURRENT_TIMESTAMP |

## Usage

```python
from ml.common.database import DatabaseClient, get_bltc_history


client = DatabaseClient.from_env()

# Use a relative window (last 3 days) for convenience.
recent_history = get_bltc_history(client, item_id=19702, last_days=3)

# Or fetch a specific calendar range (UTC recommended).
from datetime import UTC, datetime

rebuilt_history = get_bltc_history(
    client,
    item_id=19702,
    start_time=datetime(2025, 1, 1, tzinfo=UTC),
    end_time=datetime(2025, 1, 10, tzinfo=UTC),
)
```
