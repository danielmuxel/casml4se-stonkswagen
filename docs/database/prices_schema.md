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
