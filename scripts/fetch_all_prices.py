#!/usr/bin/env python3
"""
Fetch all GW2 prices from the database using parallel workers.

This script creates a snapshot of all price data up to the current time.
Each item is saved to its own CSV file in a snapshot folder.

Usage:
    python scripts/fetch_all_prices.py
    python scripts/fetch_all_prices.py --workers 12
    python scripts/fetch_all_prices.py --start-date 2024-01-01
    python scripts/fetch_all_prices.py --tradable-only
    python scripts/fetch_all_prices.py --export-tradable-items
"""

import argparse
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from gw2ml.data.retriever import DataRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch all GW2 prices from the database using parallel workers."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter in YYYY-MM-DD format (default: no lower bound)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter in YYYY-MM-DD format (default: current time)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-fetch even if cached folder exists",
    )
    parser.add_argument(
        "--tradable-only",
        action="store_true",
        help="Only fetch prices for tradable items",
    )
    parser.add_argument(
        "--export-tradable-items",
        action="store_true",
        help="Export list of all tradable items to CSV and exit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Items per query batch (default: 200). Higher = fewer queries but more memory",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between batches (default: 0.1). Higher = gentler on DB",
    )

    args = parser.parse_args()

    # Create retriever
    retriever = DataRetriever()

    # Handle --export-tradable-items separately
    if args.export_tradable_items:
        print("=" * 60)
        print("Exporting tradable items list...")
        print("=" * 60)
        df = retriever.get_tradable_items(save_csv=True)
        print(f"Exported {len(df)} tradable items")
        return

    # Parse dates
    start_time = None
    if args.start_date:
        start_time = datetime.strptime(args.start_date, "%Y-%m-%d")

    end_time = datetime.now()
    if args.end_date:
        end_time = datetime.strptime(args.end_date, "%Y-%m-%d")

    print("=" * 60)
    print("GW2 Price Data Fetcher")
    print("=" * 60)
    print(f"Workers:       {args.workers}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Batch delay:   {args.delay}s")
    print(f"Start time:    {start_time or 'No lower bound'}")
    print(f"End time:      {end_time}")
    print(f"Use cache:     {not args.no_cache}")
    print(f"Tradable only: {args.tradable_only}")
    print("=" * 60)

    output_path = retriever.get_all_prices_parallel(
        end_time=end_time,
        start_time=start_time,
        num_workers=args.workers,
        use_cache=not args.no_cache,
        tradable_only=args.tradable_only,
        batch_size=args.batch_size,
        batch_delay=args.delay,
    )

    print(f"Output folder: {output_path}")


if __name__ == "__main__":
    main()

