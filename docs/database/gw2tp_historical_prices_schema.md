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
