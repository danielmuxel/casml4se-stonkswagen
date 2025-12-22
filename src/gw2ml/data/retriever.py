import csv
import io
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import psycopg2
import requests
from sqlalchemy import create_engine, text
from massive import RESTClient
from tqdm import tqdm

# File lives under src/gw2ml/data/, so parents[3] reaches the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR_NAME = "data"


def get_data_path(
    *subpaths: Union[str, os.PathLike, Path],
    base_path: Optional[Union[str, os.PathLike, Path]] = None,
    ensure_exists: bool = True,
) -> Path:
    """
    Resolve a path relative to the configured DATA_PATH (supports absolute,
    relative, and user (~) paths) and optionally append additional subpaths.

    Args:
        subpaths: Optional additional folders or files appended in order.
        base_path: Override for DATA_PATH if provided.
        ensure_exists: Create the resulting directory if it does not exist.

    Returns:
        Absolute Path object to the resolved location.
    """
    if base_path is not None:
        base_candidate = Path(base_path)
    else:
        env_data_path = os.getenv("DATA_PATH")
        if env_data_path:
            base_candidate = Path(env_data_path)
        else:
            base_candidate = PROJECT_ROOT / DEFAULT_DATA_DIR_NAME

    base_candidate = base_candidate.expanduser()
    if not base_candidate.is_absolute():
        base_candidate = PROJECT_ROOT / base_candidate

    resolved = base_candidate
    for part in subpaths:
        if part is None:
            continue
        part_path = Path(part)
        resolved = part_path if part_path.is_absolute() else resolved / part_path

    if ensure_exists:
        resolved.mkdir(parents=True, exist_ok=True)

    return resolved


class DataRetriever:
    """Simple data retriever for crypto and stock data"""

    def __init__(self, data_folder: str = None):
        # Ensure a stable, absolute data path regardless of current working directory
        # Resolve relative paths against the project root (repo root = two levels up from src)
        # Resolve workspace-aware data paths (handles absolute + relative DATA_PATH)
        self.data_root = get_data_path(base_path=data_folder)
        self.data_folder = get_data_path("cache", base_path=self.data_root)

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

        # Map intervals to Polygon.io format
        self.massive_intervals = {
            "1m": ("minute", 1),
            "5m": ("minute", 5),
            "15m": ("minute", 15),
            "30m": ("minute", 30),
            "1h": ("hour", 1),
            "4h": ("hour", 4),
            "1d": ("day", 1),
            "1w": ("week", 1),
        }

        # Database connection string for PostgreSQL
        # Supports DB_URL directly or individual DB_USER, DB_PASSWORD, etc.
        db_url = os.getenv("DB_URL")
        
        if db_url:
            # Convert postgres:// to postgresql:// if needed (SQLAlchemy requires postgresql://)
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            db_connection_string = db_url
        else:
            # Fallback to individual env vars
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")

            if all([db_user, db_password, db_host, db_port, db_name]):
                db_connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            else:
                raise ValueError(
                    "Database not configured. Set DB_URL or individual vars (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)"
                )

        self.db_connection_string = db_connection_string
        if self.db_connection_string:
            self.db_engine = create_engine(self.db_connection_string)
        else:
            self.db_engine = None

        # Initialize Polygon.io client
        massive_api_key = os.getenv("MASSIVECOM_API_KEY")
        if massive_api_key:
            self.massive_client = RESTClient(api_key=massive_api_key)
        else:
            self.massive_client = None

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
        Get financial data

        Args:
            symbol: e.g., "BTC/USDT" for Binance, item_id for DAMU, or "AAPL" for stocks
            source: "BINANCE", "DAMU", or "STOCKS"
            start_time: datetime object
            end_time: datetime object
            interval: "1m", "5m", "15m", "30m", "1h", "1d", "1w" (for BINANCE and STOCKS)
            save_csv: automatically save to CSV
            use_cache: use cached data if available

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume (for BINANCE and STOCKS)
            or price data columns (for DAMU)
        """

        source_upper = source.upper()
        if source_upper not in ["BINANCE", "DAMU", "STOCKS"]:
            raise ValueError("Only BINANCE, DAMU, and STOCKS sources supported")

        if source_upper in ["BINANCE", "STOCKS"] and interval not in self.intervals:
            raise ValueError(f"Interval must be one of: {list(self.intervals.keys())}")

        print(f"Fetching {symbol} from {source} ({interval if source_upper in ['BINANCE', 'STOCKS'] else 'raw'})...")
        # Generate filename
        filename = f"{symbol.replace('/', '_')}_{interval if source_upper in ['BINANCE', 'STOCKS'] else 'raw'}_{
        start_time.strftime('%Y%m%d')
        }_{end_time.strftime('%Y%m%d')}.csv"
        filepath = self.data_folder / filename

        # Check cache first
        if use_cache and os.path.exists(filepath):
            print(f"Loading from cache: {filename}")
            df = pd.read_csv(filepath)
            # Convert timestamp columns based on source
            if source_upper in ["BINANCE", "STOCKS"]:
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
        elif source_upper == "STOCKS":
            df = self._fetch_stock_data(symbol, start_time, end_time, interval)

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
                     FROM prices
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


    def _fetch_stock_data(
            self, symbol: str, start_time: datetime, end_time: datetime, interval: str
    ) -> pd.DataFrame:
        """
        Fetch stock data from Polygon.io

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT", "SPY")
            start_time: start datetime
            end_time: end datetime
            interval: bar size interval

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if self.massive_client is None:
            raise ValueError("Polygon.io API key not configured. Set POLYGON_API_KEY environment variable")

        try:
            # Get massive interval
            massive_interval = self.massive_intervals.get(interval)
            if not massive_interval:
                raise ValueError(f"Invalid interval for Polygon.io: {interval}")

            timespan, multiplier = massive_interval

            # Format dates as YYYY-MM-DD
            start_date = start_time.strftime('%Y-%m-%d')
            end_date = end_time.strftime('%Y-%m-%d')

            # Fetch aggregates (bars) from Polygon.io
            aggs = []
            for agg in self.massive_client.list_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                #limit=50000
            ):
                aggs.append(agg)

            if not aggs:
                print(f"No data retrieved for {symbol}")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            } for agg in aggs])

            # Ensure timestamp is timezone-naive for consistency
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            print(f"Fetched {len(df)} bars for {symbol} from Polygon.io")

            return df

        except Exception as e:
            print(f"Error fetching data from Polygon.io: {e}")
            raise

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

    # -------------------------------------------------------------------------
    # Parallel Worker Methods for Fetching All Prices
    # -------------------------------------------------------------------------

    def get_all_item_ids(self) -> list[int]:
        """
        Fetch all distinct item_ids from the prices table.

        Returns:
            List of item_ids sorted in ascending order.
        """
        if self.db_engine is None:
            raise ValueError("Database connection not configured")

        query = text("SELECT DISTINCT item_id FROM prices ORDER BY item_id")

        with self.db_engine.connect() as connection:
            result = connection.execute(query)
            item_ids = [row[0] for row in result]

        print(f"Found {len(item_ids)} distinct item_ids")
        return item_ids

    def get_tradable_items(self, save_csv: bool = True) -> pd.DataFrame:
        """
        Fetch all tradable items from the items table.

        Args:
            save_csv: Save result to CSV file (default: True).

        Returns:
            DataFrame with all tradable items.
        """
        if self.db_engine is None:
            raise ValueError("Database connection not configured")

        query = text("""
            SELECT *
            FROM items
            WHERE is_tradeable = true
            ORDER BY id
        """)

        with self.db_engine.connect() as connection:
            df = pd.read_sql(query, connection)

        print(f"Found {len(df)} tradable items")

        if save_csv:
            output_path = self.data_folder / "tradable_items.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved tradable items to {output_path}")

        return df

    def get_tradable_item_ids(self) -> list[int]:
        """
        Fetch all tradable item_ids from the items table.

        Returns:
            List of tradable item_ids sorted in ascending order.
        """
        if self.db_engine is None:
            raise ValueError("Database connection not configured")

        query = text("""
            SELECT id
            FROM items
            WHERE is_tradeable = true
            ORDER BY id
        """)

        with self.db_engine.connect() as connection:
            result = connection.execute(query)
            item_ids = [row[0] for row in result]

        print(f"Found {len(item_ids)} tradable item_ids")
        return item_ids

    @staticmethod
    def _partition_items(item_ids: list[int], num_workers: int) -> list[list[int]]:
        """
        Split item_ids into roughly equal partitions for parallel processing.

        Args:
            item_ids: List of all item_ids to partition.
            num_workers: Number of workers/partitions to create.

        Returns:
            List of lists, each containing item_ids for one worker.
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")

        partitions = [[] for _ in range(num_workers)]
        for i, item_id in enumerate(item_ids):
            partitions[i % num_workers].append(item_id)

        return partitions

    def _fetch_items_partition(
        self,
        item_ids: list[int],
        end_time: datetime,
        worker_id: int,
        output_dir: Path,
        start_time: Optional[datetime] = None,
        progress_bar: Optional[tqdm] = None,
        progress_lock: Optional[threading.Lock] = None,
        batch_size: int = 200,
        stagger_delay: float = 0.0,
        batch_delay: float = 0.1,
    ) -> tuple[int, int, int]:
        """
        Worker function: Fetch prices for a subset of items and save each to individual CSV.

        Uses batched queries for efficiency - fetches multiple items per query,
        then splits results into individual CSV files.

        Args:
            item_ids: List of item_ids this worker is responsible for.
            end_time: Upper bound for fetched_at timestamp.
            worker_id: Identifier for this worker (for logging).
            output_dir: Directory to save individual item CSV files.
            start_time: Optional lower bound for fetched_at timestamp.
            progress_bar: Optional shared tqdm progress bar.
            progress_lock: Optional lock for thread-safe progress updates.
            batch_size: Number of items to fetch per query (default: 200).
            stagger_delay: Initial delay before starting (to stagger workers).
            batch_delay: Delay in seconds between batches (default: 0.1).

        Returns:
            Tuple of (worker_id, items_processed, total_row_count).
        """
        # Stagger worker start to avoid thundering herd
        if stagger_delay > 0:
            time.sleep(stagger_delay)

        # Create a fresh engine for this thread (thread-safe)
        # Use conservative pool settings to avoid overwhelming DB
        engine = create_engine(
            self.db_connection_string,
            pool_pre_ping=True,  # Check connection health before use
            pool_recycle=300,    # Recycle connections every 5 minutes
        )

        total_rows = 0
        items_processed = 0

        # Build base query for batch fetching
        if start_time:
            query = text("""
                SELECT id, item_id, whitelisted, buy_quantity, buy_unit_price,
                       sell_quantity, sell_unit_price, fetched_at, created_at
                FROM prices
                WHERE item_id = ANY(:item_ids)
                  AND fetched_at >= :start_time
                  AND fetched_at <= :end_time
                ORDER BY item_id, fetched_at
            """)
        else:
            query = text("""
                SELECT id, item_id, whitelisted, buy_quantity, buy_unit_price,
                       sell_quantity, sell_unit_price, fetched_at, created_at
                FROM prices
                WHERE item_id = ANY(:item_ids)
                  AND fetched_at <= :end_time
                ORDER BY item_id, fetched_at
            """)

        # Filter out items that already have CSV files (for resume support)
        remaining_items = [
            item_id for item_id in item_ids
            if not (output_dir / f"item_{item_id}.csv").exists()
        ]
        
        skipped_count = len(item_ids) - len(remaining_items)
        if skipped_count > 0 and progress_bar and progress_lock:
            with progress_lock:
                progress_bar.update(skipped_count)

        # Split remaining items into batches
        batches = [
            remaining_items[i:i + batch_size]
            for i in range(0, len(remaining_items), batch_size)
        ]

        try:
            with engine.connect() as connection:
                for batch_item_ids in batches:
                    # Build params for this batch
                    if start_time:
                        params = {
                            "item_ids": batch_item_ids,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    else:
                        params = {
                            "item_ids": batch_item_ids,
                            "end_time": end_time,
                        }

                    # Track which items we've seen data for (to handle appending)
                    items_seen: set[int] = set()

                    # Stream data in chunks to avoid RAM overflow
                    chunks = pd.read_sql(query, connection, params=params, chunksize=50_000)

                    for chunk in chunks:
                        if chunk.empty:
                            continue

                        # Convert timestamp columns
                        if 'fetched_at' in chunk.columns:
                            chunk['fetched_at'] = pd.to_datetime(chunk['fetched_at'])
                        if 'created_at' in chunk.columns:
                            chunk['created_at'] = pd.to_datetime(chunk['created_at'])

                        # Split by item_id and save/append to individual files
                        for item_id, item_df in chunk.groupby('item_id'):
                            item_file = output_dir / f"item_{item_id}.csv"

                            # First time seeing this item? Write with header and update progress
                            if item_id not in items_seen:
                                item_df.to_csv(item_file, index=False, mode='w')
                                items_seen.add(item_id)
                                items_processed += 1

                                # Update progress bar for this new item
                                if progress_bar and progress_lock:
                                    with progress_lock:
                                        progress_bar.update(1)
                            else:
                                # Append without header
                                item_df.to_csv(item_file, index=False, mode='a', header=False)

                            total_rows += len(item_df)

                    # Update progress for items in batch that had no data (empty results)
                    items_without_data = set(batch_item_ids) - items_seen
                    if items_without_data and progress_bar and progress_lock:
                        with progress_lock:
                            progress_bar.update(len(items_without_data))

                    # Delay between batches to be gentle on DB server
                    if batch_delay > 0:
                        time.sleep(batch_delay)

        except Exception as e:
            print(f"\nWorker {worker_id}: Error - {e}")
            raise

        finally:
            engine.dispose()

        return worker_id, items_processed, total_rows

    def get_all_prices_parallel(
        self,
        end_time: Optional[datetime] = None,
        start_time: Optional[datetime] = None,
        num_workers: int = 10,
        use_cache: bool = True,
        tradable_only: bool = False,
        batch_size: int = 200,
        batch_delay: float = 0.1,
    ) -> Path:
        """
        Fetch all prices from the database using parallel workers.

        Each item's data is saved to its own CSV file in a snapshot folder.
        Structure: data/cache/prices_snapshot_YYYYMMDD_HHMMSS/item_12345.csv

        Args:
            end_time: Upper bound for fetched_at (default: current time).
            start_time: Optional lower bound for fetched_at (default: no lower bound).
            num_workers: Number of parallel workers (default: 10).
            use_cache: Use cached folder if available (default: True).
            tradable_only: Only fetch tradable items (default: False).
            batch_size: Items per query batch (default: 200). Higher = fewer queries but more memory.
            batch_delay: Delay in seconds between batches (default: 0.1). Higher = gentler on DB.

        Returns:
            Path to the output folder containing individual item CSV files.
        """
        if self.db_engine is None:
            raise ValueError("Database connection not configured")

        # Default end_time to now
        if end_time is None:
            end_time = datetime.now()

        # Generate output folder name
        snapshot_time = end_time.strftime('%Y%m%d_%H%M%S')
        if start_time:
            start_str = start_time.strftime('%Y%m%d')
            folder_name = f"prices_snapshot_{start_str}_to_{snapshot_time}"
        else:
            folder_name = f"prices_snapshot_{snapshot_time}"

        if tradable_only:
            folder_name += "_tradable"

        output_dir = get_data_path(folder_name, base_path=self.data_folder, ensure_exists=True)

        # Check cache - if folder exists and has files, consider it cached
        if use_cache and output_dir.exists():
            existing_files = list(output_dir.glob("item_*.csv"))
            if existing_files:
                print(f"Loading from cache: {folder_name}/ ({len(existing_files)} items)")
                return output_dir

        print(f"Starting parallel fetch with {num_workers} workers...")
        print(f"Snapshot time: {end_time}")
        if start_time:
            print(f"Start time: {start_time}")
        print(f"Tradable only: {tradable_only}")
        print(f"Output folder: {output_dir}")

        # Step 1: Get item_ids
        start_fetch = time.time()
        if tradable_only:
            item_ids = self.get_tradable_item_ids()
        else:
            item_ids = self.get_all_item_ids()

        if not item_ids:
            print("No items found in database")
            return output_dir

        # Step 2: Partition items across workers
        partitions = self._partition_items(item_ids, num_workers)
        print(f"Partitioned {len(item_ids):,} items across {num_workers} workers\n")

        # Step 3: Execute workers in parallel with progress bar
        total_rows = 0
        total_items = 0
        failed_workers = []

        # Create shared progress bar and lock
        progress_lock = threading.Lock()
        progress_bar = tqdm(
            total=len(item_ids),
            desc="Fetching items",
            unit="items",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        # Stagger workers to avoid thundering herd on DB
        stagger_interval = 0.2  # 200ms between each worker start

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_items_partition,
                    partition,
                    end_time,
                    worker_id,
                    output_dir,
                    start_time,
                    progress_bar,
                    progress_lock,
                    batch_size,
                    worker_id * stagger_interval,  # Stagger start: worker 0 = 0s, worker 1 = 0.2s, etc.
                    batch_delay,
                ): worker_id
                for worker_id, partition in enumerate(partitions)
            }

            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    w_id, items_processed, row_count = future.result()
                    total_items += items_processed
                    total_rows += row_count
                except Exception as e:
                    print(f"\nWorker {worker_id} failed: {e}")
                    failed_workers.append(worker_id)

        progress_bar.close()

        if failed_workers:
            print(f"\nWarning: {len(failed_workers)} workers failed: {failed_workers}")

        elapsed = time.time() - start_fetch
        print("=" * 60)
        print(f"Completed in {elapsed:.2f}s")
        print(f"Total items: {total_items:,}")
        print(f"Total rows: {total_rows:,}")
        print(f"Output folder: {output_dir}")
        print("=" * 60)

        return output_dir