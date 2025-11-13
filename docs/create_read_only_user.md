# Create a Read-Only User for `gw2_trading_prices_history`

## Overview

This guide walks through creating (or refreshing) a PostgreSQL login that can only read data in the `gw2_trading_prices_history` database. The SQL snippets below are safe to run repeatedly: they create the `readonly_user` role if it does not exist, reset its password if it already does, and ensure it keeps read-only access to current and future database objects.

## Prerequisites

- Access to `psql` (or another PostgreSQL client) as a superuser or a role that can manage roles and grant privileges.
- The `gw2_trading_prices_history` database already exists.
- A strong password ready to replace the `'strong_password'` placeholder.

> **Tip:** If you prefer a different role or database name, replace every occurrence of `readonly_user` or `gw2_trading_prices_history` before running the commands.

## How to Run

1. Update the password in the SQL snippets below.
2. Paste each snippet into an interactive `psql` session connected to the target database **or** save the snippets to a `.sql` file and execute it:
   ```
   psql -d gw2_trading_prices_history -f path/to/create_read_only_user.sql
   ```

## Step-by-Step SQL

### 1. Create or Update the Login Role

Ensures the `readonly_user` login exists and has a strong password.

```sql
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'readonly_user') THEN
    CREATE ROLE readonly_user
      LOGIN
      NOSUPERUSER
      NOCREATEDB
      NOCREATEROLE
      NOREPLICATION
      INHERIT
      PASSWORD 'strong_password';
  ELSE
    ALTER ROLE readonly_user LOGIN PASSWORD 'strong_password';
  END IF;
END$$;
```

### 2. Allow Database Connections

Grants the role permission to connect to the `gw2_trading_prices_history` database.

```sql
GRANT CONNECT ON DATABASE gw2_trading_prices_history TO readonly_user;
```

### 3. Apply Read-Only Privileges to All Schemas

Loops through every non-system schema so the role can query existing tables and sequences while being explicitly denied write operations. It also sets default privileges so future tables and sequences stay read-only.

```sql
DO $$
DECLARE
  schema_name text;
BEGIN
  FOR schema_name IN
    SELECT nspname
    FROM pg_namespace
    WHERE nspname NOT LIKE 'pg_%'
      AND nspname <> 'information_schema'
  LOOP
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO %I', schema_name, 'readonly_user');
    EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO %I', schema_name, 'readonly_user');
    EXECUTE format('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA %I TO %I', schema_name, 'readonly_user');

    EXECUTE format('REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA %I FROM %I', schema_name, 'readonly_user');
    EXECUTE format('REVOKE UPDATE ON ALL SEQUENCES IN SCHEMA %I FROM %I', schema_name, 'readonly_user');

    EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA %I GRANT SELECT ON TABLES TO %I', schema_name, 'readonly_user');
    EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA %I GRANT USAGE, SELECT ON SEQUENCES TO %I', schema_name, 'readonly_user');
  END LOOP;
END$$;
```