"""
Scan all tradeable TP items as-of a fixed timestamp and rank opportunities.

Outputs:
- most volatile items over a lookback window
- biggest profit after Trading Post fee (default 15%)
- a filtered list where profit_after_fee > 0

Usage (examples):
  uv run --group dev python scripts/scan_tp_opportunities.py --as-of "2025-12-18T12:00:00Z" --lookback-hours 24
  uv run --group dev python scripts/scan_tp_opportunities.py --top-n 50 --output-csv data/opportunities.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from gw2ml import DatabaseClient
from gw2ml.evaluation.market_scan import TpMarketScanConfig, build_tp_opportunity_table
from gw2ml.evaluation.price_file_scan import PricesFileScanConfig, build_opportunity_table_from_prices_dir


def _parse_datetime_utc(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _infer_as_of_from_snapshot_dir(prices_dir: Path) -> datetime | None:
    """
    Infer `as_of` from a snapshot directory name like `20251204T161148Z`.
    """

    name = Path(prices_dir).name.strip()
    if len(name) != 16 or not name.endswith("Z") or "T" not in name:
        return None
    try:
        dt = datetime.strptime(name, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC)


def _format_table(df: pd.DataFrame, *, cols: list[str], top_n: int) -> str:
    view = df.loc[:, [c for c in cols if c in df.columns]].head(top_n)
    return view.to_string(index=False, justify="left")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan TP items for volatility and profit-after-fee opportunities.")
    parser.add_argument("--as-of", type=str, default=None, help="UTC timestamp for analysis, e.g. 2025-12-18T12:00:00Z")
    parser.add_argument("--lookback-hours", type=int, default=24, help="Lookback window size (hours) for volatility stats.")
    parser.add_argument("--fee-rate", type=float, default=0.15, help="Trading Post fee rate as fraction (default: 0.15).")
    parser.add_argument("--min-points", type=int, default=30, help="Min points required for volatility computation.")
    parser.add_argument("--top-n", type=int, default=25, help="Number of rows to print for each ranking.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to write the full opportunity table CSV.")
    parser.add_argument("--db-url", type=str, default=None, help="Optional DB URL override (otherwise uses DB_URL env/streamlit secrets).")
    parser.add_argument(
        "--prices-dir",
        type=str,
        default=None,
        help="Optional directory containing per-item exports like data/20251204T161148Z/prices_<id>_<snapshot>.csv",
    )
    parser.add_argument("--workers", type=int, default=8, help="Worker threads for --prices-dir scanning.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for number of price CSV files to scan.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--log-every", type=int, default=1000, help="Log progress every N files in --prices-dir mode.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("scan_tp_opportunities")
    started_at = time.time()

    if args.as_of:
        as_of = _parse_datetime_utc(args.as_of)
    elif args.prices_dir:
        inferred = _infer_as_of_from_snapshot_dir(Path(args.prices_dir))
        as_of = inferred if inferred is not None else datetime.now(UTC)
    else:
        as_of = datetime.now(UTC)
    lookback = timedelta(hours=max(int(args.lookback_hours), 1))
    fee_rate = float(args.fee_rate)
    min_points = int(args.min_points)

    if args.prices_dir:
        logger.info("Mode: offline CSV scan")
        logger.info("Prices dir: %s", args.prices_dir)
        config = PricesFileScanConfig(
            prices_dir=Path(args.prices_dir),
            as_of=as_of,
            lookback=lookback,
            fee_rate=fee_rate,
            min_points=min_points,
            max_files=args.max_files,
        )
        table = build_opportunity_table_from_prices_dir(
            config=config,
            workers=int(args.workers),
            log_every=int(args.log_every),
        )
        if table.empty:
            print("No data returned from prices directory scan. Check path / as-of / lookback.")
            return 2
    else:
        logger.info("Mode: database scan")
        config = TpMarketScanConfig(
            as_of=as_of,
            lookback=lookback,
            fee_rate=fee_rate,
            min_points=min_points,
        )
        client = DatabaseClient.from_env(db_url_override=args.db_url)
        table = build_tp_opportunity_table(client, config=config)
        if table.empty:
            print("No data returned (empty snapshot). Check DB connection and timestamp range.")
            return 2

        # Normalize column names to match the offline scan output.
        table = table.rename(
            columns={
                "buy_price": "buy_price",
                "sell_price": "sell_price",
                "timestamp_utc": "snapshot_time",
            }
        )
    if table.empty:
        print("No data returned.")
        return 2

    top_n = max(int(args.top_n), 1)
    cols = [
        "item_id",
        "buy_price",
        "sell_price",
        "profit_after_fee",
        "roi_after_fee",
        "return_std",
        "mid_cv",
        "points",
        "snapshot_time",
    ]

    most_volatile = table.sort_values(["return_std", "mid_cv"], ascending=[False, False], na_position="last")
    biggest_profit = table.sort_values(["profit_after_fee", "roi_after_fee"], ascending=[False, False], na_position="last")
    profitable = table[table["is_profitable_after_fee"] == True].sort_values(  # noqa: E712
        ["profit_after_fee", "roi_after_fee"],
        ascending=[False, False],
        na_position="last",
    )

    print(f"Scan as-of: {as_of.isoformat()} | lookback: {lookback} | fee: {fee_rate:.2%}")
    print()
    print("== Most volatile (return_std) ==")
    print(_format_table(most_volatile, cols=cols, top_n=top_n))
    print()
    print("== Biggest profit after fee (per unit) ==")
    print(_format_table(biggest_profit, cols=cols, top_n=top_n))
    print()
    print("== Profitable after fee (profit_after_fee > 0) ==")
    print(_format_table(profitable, cols=cols, top_n=top_n))

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        print()
        print(f"Wrote CSV: {output_path}")

    logger.info("Done in %.2fs", time.time() - started_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


