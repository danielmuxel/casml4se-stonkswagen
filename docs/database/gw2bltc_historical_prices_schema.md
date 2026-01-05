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
