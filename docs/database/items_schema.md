# items Schema

| column_name | data_type | is_nullable | column_default |
| --- | --- | --- | --- |
| id | integer | NO |  |
| chat_link | character varying | NO |  |
| name | character varying | NO |  |
| icon | character varying | YES |  |
| description | text | YES |  |
| type | character varying | NO |  |
| rarity | character varying | NO |  |
| level | integer | NO |  |
| vendor_value | integer | NO |  |
| default_skin | integer | YES |  |
| flags | ARRAY | YES | '{}'::text[] |
| game_types | ARRAY | YES | '{}'::text[] |
| restrictions | ARRAY | YES | '{}'::text[] |
| is_tradeable | boolean | YES | true |
| upgrades_into | jsonb | YES |  |
| upgrades_from | jsonb | YES |  |
| details | jsonb | YES |  |
| created_at | timestamp with time zone | YES | CURRENT_TIMESTAMP |
| updated_at | timestamp with time zone | YES | CURRENT_TIMESTAMP |
