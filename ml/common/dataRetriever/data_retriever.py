import os
import time
from datetime import datetime

import pandas as pd
import requests
import sqlalchemy
from sqlalchemy import create_engine, text

class DataRetriever:
    """Simple data retriever for crypto data"""

    def __init__(self, data_folder: str = "../data/cache/"):
        # Ensure a stable, absolute data path regardless of current working directory
        # Resolve relative paths against the project root (repo root = two levels up from src)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))

        if data_folder is None:
            resolved_folder = os.path.join(project_root, "data")
        elif os.path.isabs(data_folder):
            resolved_folder = data_folder
        else:
            resolved_folder = os.path.join(project_root, data_folder)

        self.data_folder = os.path.normpath(resolved_folder)
        os.makedirs(self.data_folder, exist_ok=True)

        # Binance settings
        self.binance_url = "https://api.binance.com/api/v3/klines"
        self.intervals = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w",
        }
        # Database connection string for PostgreSQL
        # Format: postgresql://username:password@host:port/database
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

        if all([db_user, db_password, db_host, db_port, db_name]):
            db_connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            db_connection_string = None
            raise ValueError # TODO throw error message

        self.db_connection_string = db_connection_string
        if self.db_connection_string:
            self.db_engine = create_engine(self.db_connection_string)
        else:
            self.db_engine = None

    def get_data(
            self,
            symbol: str,
            source: str,
            start_time: datetime,
            end_time: datetime,
            interval: str,
            save_csv: bool = True,
            use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get crypto data

        Args:
            symbol: e.g., "BTC/USDT" for Binance or item_id for DAMU
            source: "BINANCE" or "DAMU"
            start_time: datetime object
            end_time: datetime object
            interval: "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w" (only for BINANCE)
            save_csv: automatically save to CSV

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume (for BINANCE)
            or price data columns (for DAMU)
        """

        source_upper = source.upper()
        if source_upper not in ["BINANCE", "DAMU"]:
            raise ValueError("Only BINANCE and DAMU sources supported")

        if source_upper == "BINANCE" and interval not in self.intervals:
            raise ValueError(f"Interval must be one of: {list(self.intervals.keys())}")

        print(f"Fetching {symbol} from {source} ({interval if source_upper == 'BINANCE' else 'raw'})...")
        # Generate filename
        filename = f"{symbol.replace('/', '_')}_{interval if source_upper == 'BINANCE' else 'raw'}_{
        start_time.strftime('%Y%m%d')
        }_{end_time.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_folder, filename)

        # Check cache first
        if use_cache and os.path.exists(filepath):
            print(f"Loading from cache: {filename}")
            df = pd.read_csv(filepath)
            # Convert timestamp columns based on source
            if source_upper == "BINANCE":
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            elif source_upper == "DAMU":
                if 'fetched_at' in df.columns:
                    df['fetched_at'] = pd.to_datetime(df['fetched_at'])
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                # Ensure a consistent 'timestamp' column for downstream consumers
                if 'timestamp' not in df.columns and 'fetched_at' in df.columns:
                    df['timestamp'] = df['fetched_at']
            return df

        # Fetch new data based on source
        if source_upper == "BINANCE":
            df = self._fetch_binance_data(symbol, start_time, end_time, interval)
        elif source_upper == "DAMU":
            df = self._fetch_damu_price(symbol, start_time, end_time)
            if 'fetched_at' in df.columns and 'timestamp' not in df.columns:
                df['timestamp'] = df['fetched_at']

        # Save to CSV
        if save_csv:
            df.to_csv(filepath, index=False)
            print(f"Saved to {filepath}")

        return df

    def _fetch_damu_price(
            self, symbol: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch price data from PostgreSQL database

        Args:
            symbol: item identifier (item_id)
            start_time: start datetime
            end_time: end datetime

        Returns:
            DataFrame with price data
        """
        if self.db_engine is None:
            raise ValueError("Database connection not configured. Pass db_connection_string to __init__")

        # Convert symbol to item_id (assuming symbol is the item_id or needs conversion)
        try:
            item_id = int(symbol)
        except ValueError:
            raise ValueError(f"Symbol must be convertible to item_id (integer), got: {symbol}")

        # SQL query to fetch prices within the time range
        query = text("""
                     SELECT id,
                            item_id,
                            whitelisted,
                            buy_quantity,
                            buy_unit_price,
                            sell_quantity,
                            sell_unit_price,
                            fetched_at,
                            created_at
                     FROM public.prices
                     WHERE item_id = :item_id
                       AND fetched_at >= :start_time
                       AND fetched_at <= :end_time
                     ORDER BY fetched_at ASC
                     """)

        # Execute query and load into DataFrame
        with self.db_engine.connect() as connection:
            df = pd.read_sql(
                query,
                connection,
                params={
                    "item_id": item_id,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )

        print(f"Fetched {len(df)} price records for item_id {item_id}")

        # Convert timestamp columns to datetime
        if 'fetched_at' in df.columns:
            df['fetched_at'] = pd.to_datetime(df['fetched_at'])
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])

        return df



    def _fetch_binance_data(
        self, symbol: str, start_time: datetime, end_time: datetime, interval: str
    ) -> pd.DataFrame:
        """Fetch data from Binance API"""

        # Convert symbol format: BTC/USDT -> BTCUSDT
        binance_symbol = symbol.replace("/", "")

        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        all_data = []
        current_start = start_ms

        while current_start < end_ms:
            params = {
                "symbol": binance_symbol,
                "interval": self.intervals[interval],
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }

            response = requests.get(self.binance_url, params=params)
            response.raise_for_status()

            data = response.json()
            if not data:
                break

            all_data.extend(data)
            current_start = data[-1][6] + 1  # Next start time
            time.sleep(0.1)  # Rate limiting

        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades_count",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )

        # Clean up
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)

        return df[["timestamp", "open", "high", "low", "close", "volume"]]
