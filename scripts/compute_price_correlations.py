#!/usr/bin/env python3
"""
Compute item-to-item correlations and clusters from cached price snapshots.

Outputs (per mode):
- Atemporal (distributional features): top_pairs_atemporal.csv, clusters_atemporal.csv,
  optional correlations_items_atemporal.csv, features_atemporal.csv
- Temporal (time-aligned): top_pairs_temporal.csv, clusters_temporal.csv,
  optional correlations_items_temporal.csv

Design:
- Streams per-item CSVs; uses sell_unit_price for similarity.
- Temporal mode resamples to a configurable frequency (choices: 5min, 1H, 4H, 1D).
- Clusters are connected components of a graph built from edges with |corr| >= threshold.
- Logging is verbose to show progress and phases.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def find_latest_snapshot(base_dir: Path) -> Path:
    pattern_variants = ["prices_snapshot_*", "price_snapshot_*"]
    candidates: List[Path] = []
    for pattern in pattern_variants:
        candidates.extend(base_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No snapshot directories found in {base_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_item_id(path: Path) -> str:
    name = path.stem  # item_<id>
    if name.startswith("item_"):
        return name.split("_", 1)[1]
    return name


def list_item_files(snapshot_dir: Path) -> List[Path]:
    files = sorted(snapshot_dir.glob("item_*.csv"))
    if not files:
        raise FileNotFoundError(f"No item_*.csv files found in {snapshot_dir}")
    return files


def compute_atemporal_features(
    files: Sequence[Path],
    price_col: str,
    chunk_size: int,
    min_pairs: int,
    logger: logging.Logger,
) -> Tuple[List[str], np.ndarray]:
    item_ids: List[str] = []
    feats: List[List[float]] = []
    for idx, path in enumerate(files, 1):
        vals: List[float] = []
        for chunk in pd.read_csv(path, usecols=[price_col], chunksize=chunk_size):
            series = chunk[price_col].dropna()
            if not series.empty:
                vals.append(series.to_numpy(dtype=np.float64, copy=False))
        if not vals:
            continue
        arr = np.concatenate(vals)
        count = arr.size
        if count < min_pairs:
            continue
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if count > 1 else 0.0
        if std == 0:
            continue
        median = float(np.median(arr))
        p10 = float(np.percentile(arr, 10))
        p90 = float(np.percentile(arr, 90))
        item_ids.append(extract_item_id(path))
        feats.append([count, mean, std, median, p10, p90])
        if idx % 1000 == 0 or idx == len(files):
            logger.info("Atemporal features: processed %d/%d files", idx, len(files))
    if not feats:
        raise ValueError("No atemporal features computed; check data or min-pairs setting.")
    features = np.asarray(feats, dtype=np.float32)
    return item_ids, features


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, ddof=1, keepdims=True)
    std[std == 0] = np.nan
    z = (matrix - mean) / std
    return z


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:n
        comp_ids: List[int] = [0] * len(self.parent)
        comp_sizes: Dict[int, int] = {}
        next_id = 0
        for i in range(len(self.parent)):
            r = self.find(i)
            if r not in roots:
                roots[r] = next_id
                comp_sizes[next_id] = self.size[r]
                next_id += 1
            comp_ids[i] = roots[r]
        return comp_ids, comp_sizes


def atemporal_correlations(
    item_ids: List[str],
    features: np.ndarray,
    threshold: float,
    top_k: int,
    save_full: bool,
    snapshot_dir: Path,
    logger: logging.Logger,
    block_size: int,
    max_pairs_store: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_items, n_feats = features.shape
    logger.info("Atemporal: %d items, %d features each", n_items, n_feats)
    z = zscore_rows(features)
    valid_rows = ~np.isnan(z).any(axis=1)
    if not np.all(valid_rows):
        logger.info("Atemporal: dropping %d rows with NaN zscores", (~valid_rows).sum())
        z = z[valid_rows]
        item_ids = [item_ids[i] for i, ok in enumerate(valid_rows) if ok]
        n_items = len(item_ids)
    logger.info("Atemporal corr: starting blocks of size %d", block_size)
    dsu = DisjointSet(n_items)
    records_heap: List[Tuple[float, float, int, int]] = []  # (abs_corr, corr, i, j)
    total_edges = 0
    block = max(1, block_size)
    for start in range(0, n_items, block):
        end = min(n_items, start + block)
        z_block = z[start:end]
        corr_block = (z_block @ z.T) / (n_feats - 1)
        for i in range(corr_block.shape[0]):
            for j in range(start + i + 1, n_items):
                c = corr_block[i, j]
                if abs(c) >= threshold:
                    total_edges += 1
                    dsu.union(start + i, j)
                    abs_c = float(abs(c))
                    entry = (abs_c, float(c), start + i, j)
                    if len(records_heap) < max_pairs_store:
                        heapq.heappush(records_heap, entry)
                    elif abs_c > records_heap[0][0]:
                        heapq.heapreplace(records_heap, entry)
        if (start // block) % 10 == 0 or end == n_items:
            logger.info("Atemporal corr: processed rows %d/%d", end, n_items)
    logger.info("Atemporal corr: edges above threshold=%s: %d (stored up to %d)", threshold, total_edges, max_pairs_store)

    comp_ids, comp_sizes = dsu.components()

    clusters_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "cluster_id": comp_ids,
            "cluster_size": [comp_sizes[c] for c in comp_ids],
        }
    ).sort_values(["cluster_size", "cluster_id", "item_id"], ascending=[False, True, True])

    records_sorted = sorted(records_heap, key=lambda x: x[0], reverse=True)
    pairs_df = pd.DataFrame(
        {
            "item_a": [item_ids[i] for _, _, i, _ in records_sorted],
            "item_b": [item_ids[j] for _, _, _, j in records_sorted],
            "corr": [c for _, c, _, _ in records_sorted],
        }
    )
    if not pairs_df.empty:
        pairs_df["abs_corr"] = pairs_df["corr"].abs()
    if top_k and len(pairs_df) > top_k:
        logger.info("Atemporal: top %d of %d pairs retained for display", top_k, len(pairs_df))

    corr_df = pd.DataFrame()
    if save_full:
        logger.info("Atemporal: saving full correlation matrix")
        corr_full = np.eye(n_items, dtype=np.float32)
        # Recompute full matrix in blocks for writing
        for start in range(0, n_items, block):
            end = min(n_items, start + block)
            z_block = z[start:end]
            corr_full[start:end] = (z_block @ z.T) / (n_feats - 1)
        corr_df = pd.DataFrame(corr_full, index=item_ids, columns=item_ids)
        corr_df.to_csv(snapshot_dir / "correlations_items_atemporal.csv")

    pairs_df.to_csv(snapshot_dir / "top_pairs_atemporal.csv", index=False)
    clusters_df.to_csv(snapshot_dir / "clusters_atemporal.csv", index=False)
    return clusters_df, pairs_df, corr_df


def collect_temporal_data(
    files: Sequence[Path],
    time_col: str,
    price_col: str,
    freq: str,
    chunk_size: int,
    min_pairs: int,
    logger: logging.Logger,
) -> Tuple[List[str], np.ndarray, List[pd.Timestamp]]:
    """
    Single-pass collection: reads each file once, collects aggregated prices per date,
    then builds the time matrix. Returns (item_ids, matrix, date_index).
    """
    # Store per-item aggregated data: item_id -> {date -> [prices]}
    item_data: Dict[str, Dict[pd.Timestamp, List[float]]] = {}
    date_set: set = set()

    for idx, path in enumerate(files, 1):
        item_id = extract_item_id(path)
        date_prices: Dict[pd.Timestamp, List[float]] = {}
        row_count = 0

        for chunk in pd.read_csv(path, usecols=[time_col, price_col], chunksize=chunk_size):
            chunk = chunk.dropna(subset=[price_col, time_col])
            if chunk.empty:
                continue
            chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
            chunk = chunk.dropna(subset=[time_col])
            if chunk.empty:
                continue
            chunk["__dt"] = chunk[time_col].dt.floor(freq)
            row_count += len(chunk)
            for dt, grp in chunk.groupby("__dt"):
                date_prices.setdefault(dt, []).extend(grp[price_col].tolist())

        # Only keep items with enough data points
        if row_count >= min_pairs and date_prices:
            item_data[item_id] = date_prices
            date_set.update(date_prices.keys())

        if idx % 1000 == 0 or idx == len(files):
            logger.info("Temporal collection: processed %d/%d files, items kept: %d, unique dates: %d",
                       idx, len(files), len(item_data), len(date_set))

    if not item_data:
        raise ValueError("No items with sufficient data for temporal analysis")

    # Build date index and matrix
    date_index = sorted(date_set)
    date_to_idx = {dt: i for i, dt in enumerate(date_index)}
    item_ids = sorted(item_data.keys())
    n_items = len(item_ids)
    n_dates = len(date_index)

    logger.info("Building matrix: %d items x %d dates", n_items, n_dates)
    matrix = np.full((n_items, n_dates), np.nan, dtype=np.float32)

    for item_pos, item_id in enumerate(item_ids):
        for dt, prices in item_data[item_id].items():
            d_idx = date_to_idx[dt]
            matrix[item_pos, d_idx] = float(np.mean(prices))

    # Clear memory
    del item_data

    return item_ids, matrix, date_index


def temporal_correlations(
    item_ids: List[str],
    matrix: np.ndarray,
    threshold: float,
    min_pairs: int,
    top_k: int,
    save_full: bool,
    snapshot_dir: Path,
    logger: logging.Logger,
    block_size: int,
    max_pairs_store: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_items, n_dates = matrix.shape
    logger.info("Temporal: matrix %d items x %d dates", n_items, n_dates)
    mask = ~np.isnan(matrix)
    counts = mask.sum(axis=1)
    valid_mask = counts >= min_pairs
    if not np.all(valid_mask):
        logger.info("Temporal: dropping %d items with < min_pairs (%d)", (~valid_mask).sum(), min_pairs)
        matrix = matrix[valid_mask]
        mask = mask[valid_mask]
        item_ids = [item_ids[i] for i, ok in enumerate(valid_mask) if ok]
        n_items = len(item_ids)
    # stats
    sums = np.nansum(matrix, axis=1)
    n_obs = mask.sum(axis=1)
    means = sums / n_obs
    centered = matrix - means[:, None]
    centered[~mask] = 0.0
    sumsq = np.nansum(centered * centered, axis=1)
    stds = np.sqrt(sumsq / np.maximum(n_obs - 1, 1))
    stds[stds == 0] = np.nan

    dsu = DisjointSet(n_items)
    records_heap: List[Tuple[float, float, int, int, int]] = []  # (abs_corr, corr, i, j, n_ij)
    total_edges = 0

    block = max(1, block_size)
    for start in range(0, n_items, block):
        end = min(n_items, start + block)
        c_block = centered[start:end]
        m_block = mask[start:end].astype(np.float32)
        counts_block = m_block @ mask.astype(np.float32).T
        numer = c_block @ centered.T
        denom = (counts_block - 1) * (stds[start:end][:, None] * stds[None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_block = numer / denom
        for i in range(corr_block.shape[0]):
            for j in range(start + i + 1, n_items):
                n_ij = counts_block[i, j]
                if n_ij < min_pairs:
                    continue
                c = corr_block[i, j]
                if np.isnan(c):
                    continue
                if abs(c) >= threshold:
                    total_edges += 1
                    dsu.union(start + i, j)
                    abs_c = float(abs(c))
                    entry = (abs_c, float(c), start + i, j, int(n_ij))
                    if len(records_heap) < max_pairs_store:
                        heapq.heappush(records_heap, entry)
                    elif abs_c > records_heap[0][0]:
                        heapq.heapreplace(records_heap, entry)
        logger.info("Temporal corr: processed rows %d/%d", end, n_items)

    logger.info("Temporal corr: edges above threshold=%s: %d (stored up to %d)", threshold, total_edges, max_pairs_store)
    comp_ids, comp_sizes = dsu.components()

    clusters_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "cluster_id": comp_ids,
            "cluster_size": [comp_sizes[c] for c in comp_ids],
        }
    ).sort_values(["cluster_size", "cluster_id", "item_id"], ascending=[False, True, True])

    records_sorted = sorted(records_heap, key=lambda x: x[0], reverse=True)
    pairs_df = pd.DataFrame(
        {
            "item_a": [item_ids[i] for _, _, i, _, _ in records_sorted],
            "item_b": [item_ids[j] for _, _, _, j, _ in records_sorted],
            "corr": [c for _, c, _, _, _ in records_sorted],
            "pair_count": [n for _, _, _, _, n in records_sorted],
        }
    )
    if not pairs_df.empty:
        pairs_df["abs_corr"] = pairs_df["corr"].abs()

    corr_df = pd.DataFrame()
    if save_full:
        logger.info("Temporal: saving full correlation matrix in blocks")
        corr_full = np.eye(n_items, dtype=np.float32)
        for start in range(0, n_items, block):
            end = min(n_items, start + block)
            c_block = centered[start:end]
            m_block = mask[start:end].astype(np.float32)
            counts_block = m_block @ mask.astype(np.float32).T
            numer = c_block @ centered.T
            denom = (counts_block - 1) * (stds[start:end][:, None] * stds[None, :])
            with np.errstate(invalid="ignore", divide="ignore"):
                corr_full[start:end] = numer / denom
            logger.info("Temporal full matrix: wrote rows %d-%d", start, end)
        corr_df = pd.DataFrame(corr_full, index=item_ids, columns=item_ids)
        corr_df.to_csv(snapshot_dir / "correlations_items_temporal.csv")

    pairs_df.to_csv(snapshot_dir / "top_pairs_temporal.csv", index=False)
    clusters_df.to_csv(snapshot_dir / "clusters_temporal.csv", index=False)
    return clusters_df, pairs_df, corr_df


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute item correlations and clusters for price snapshots.")
    parser.add_argument("--snapshot", type=Path, default=None, help="Specific snapshot dir. Defaults to latest in data/cache.")
    parser.add_argument("--freq", type=str, default="1h", choices=["5min", "1h", "4h", "1D"], help="Temporal resample frequency.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Absolute correlation threshold for edges/clusters.")
    parser.add_argument("--top-k", type=int, default=50, help="Top pairs to display (files always contain all above threshold).")
    parser.add_argument("--min-pairs", type=int, default=10, help="Minimum overlapping observations per pair/item.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="CSV chunk size.")
    parser.add_argument("--atemporal-block-size", type=int, default=256, help="Block size for atemporal correlation computation.")
    parser.add_argument("--temporal-block-size", type=int, default=256, help="Block size for temporal correlation computation.")
    parser.add_argument("--atemporal-only", action="store_true", help="Run only the atemporal pass.")
    parser.add_argument("--temporal-only", action="store_true", help="Run only the temporal pass.")
    parser.add_argument("--max-pairs-store", type=int, default=200000, help="Max pairs kept in memory (per mode) for top-pairs output.")
    parser.add_argument("--save-full-atemporal", action="store_true", help="Save full item-item atemporal correlation matrix.")
    parser.add_argument("--save-full-temporal", action="store_true", help="Save full item-item temporal correlation matrix.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--price-col", type=str, default="sell_unit_price", help="Price column to use for similarity.")
    # Testing/subset options
    parser.add_argument("--max-items", type=int, default=None, help="Limit to first N items for testing (default: all).")
    parser.add_argument("--sample-items", type=int, default=None, help="Randomly sample N items for testing (default: all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample-items (default: 42).")
    return parser.parse_args(argv)


def apply_item_limits(
    files: List[Path],
    max_items: int | None,
    sample_items: int | None,
    seed: int,
    logger: logging.Logger,
) -> List[Path]:
    """Apply --max-items or --sample-items filtering to the file list."""
    if sample_items is not None and sample_items < len(files):
        random.seed(seed)
        files = random.sample(files, sample_items)
        logger.info("Randomly sampled %d items (seed=%d)", sample_items, seed)
    elif max_items is not None and max_items < len(files):
        files = files[:max_items]
        logger.info("Limited to first %d items", max_items)
    return files


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    logger = logging.getLogger("correlations")

    base_dir = Path("data/cache")
    snapshot_dir = args.snapshot or find_latest_snapshot(base_dir)
    logger.info("Using snapshot directory: %s", snapshot_dir)

    files = list_item_files(snapshot_dir)
    logger.info("Found %d item files", len(files))

    # Apply item limits for testing
    files = apply_item_limits(files, args.max_items, args.sample_items, args.seed, logger)

    run_atemporal = True
    run_temporal = True
    if args.atemporal_only and args.temporal_only:
        logger.warning("Both --atemporal-only and --temporal-only set; running both passes.")
    elif args.atemporal_only:
        run_temporal = False
    elif args.temporal_only:
        run_atemporal = False

    at_pairs_df = pd.DataFrame()
    t_pairs_df = pd.DataFrame()

    if run_atemporal:
        logger.info("=== Atemporal pass ===")
        at_item_ids, features = compute_atemporal_features(
            files=files,
            price_col=args.price_col,
            chunk_size=args.chunk_size,
            min_pairs=args.min_pairs,
            logger=logger,
        )
        features_df = pd.DataFrame(
            features,
            columns=["count", "mean", "std", "median", "p10", "p90"],
        )
        features_df.insert(0, "item_id", at_item_ids)
        features_df.to_csv(snapshot_dir / "features_atemporal.csv", index=False)
        at_clusters_df, at_pairs_df, _ = atemporal_correlations(
            item_ids=at_item_ids,
            features=features,
            threshold=args.threshold,
            top_k=args.top_k,
            save_full=args.save_full_atemporal,
            snapshot_dir=snapshot_dir,
            logger=logger,
            block_size=args.atemporal_block_size,
        max_pairs_store=args.max_pairs_store,
        )
        logger.info(
            "Atemporal done: %d clusters, %d pairs >= threshold",
            at_clusters_df["cluster_id"].nunique(),
            len(at_pairs_df),
        )

    if run_temporal:
        logger.info("=== Temporal pass ===")
        temp_item_ids, matrix, dates = collect_temporal_data(
            files=files,
            time_col="fetched_at",
            price_col=args.price_col,
            freq=args.freq,
            chunk_size=args.chunk_size,
            min_pairs=args.min_pairs,
            logger=logger,
        )
        logger.info("Temporal: %d dates after resample to %s, %d items with enough data", len(dates), args.freq, len(temp_item_ids))
        t_clusters_df, t_pairs_df, _ = temporal_correlations(
            item_ids=temp_item_ids,
            matrix=matrix,
            threshold=args.threshold,
            min_pairs=args.min_pairs,
            top_k=args.top_k,
            save_full=args.save_full_temporal,
            snapshot_dir=snapshot_dir,
            logger=logger,
            block_size=args.temporal_block_size,
        max_pairs_store=args.max_pairs_store,
        )
        logger.info(
            "Temporal done: %d clusters, %d pairs >= threshold",
            t_clusters_df["cluster_id"].nunique(),
            len(t_pairs_df),
        )

    logger.info("Finished. Outputs written to %s", snapshot_dir)
    if args.top_k > 0:
        if run_atemporal and not at_pairs_df.empty:
            print("\nTop atemporal pairs:")
            print(at_pairs_df.head(args.top_k).to_string(index=False))
        if run_temporal and not t_pairs_df.empty:
            print("\nTop temporal pairs:")
            print(t_pairs_df.head(args.top_k).to_string(index=False))

    # Print cluster summaries
    if run_atemporal:
        print_cluster_summary(snapshot_dir / "clusters_atemporal.csv", "Atemporal", args.top_k)
    if run_temporal:
        print_cluster_summary(snapshot_dir / "clusters_temporal.csv", "Temporal", args.top_k)

    return 0


def print_cluster_summary(clusters_path: Path, mode: str, top_k: int) -> None:
    """Print a summary of clusters grouped by cluster_id."""
    if not clusters_path.exists():
        return

    df = pd.read_csv(clusters_path)
    if df.empty:
        return

    # Group by cluster and get stats
    cluster_groups = df.groupby("cluster_id").agg(
        size=("item_id", "count"),
        items=("item_id", lambda x: list(x)),
    ).reset_index()
    cluster_groups = cluster_groups.sort_values("size", ascending=False)

    # Filter to clusters with >1 item (singletons aren't interesting)
    multi_item_clusters = cluster_groups[cluster_groups["size"] > 1]

    if multi_item_clusters.empty:
        print(f"\n{mode} clusters: No multi-item clusters found.")
        return

    print(f"\n{mode} clusters (size > 1): {len(multi_item_clusters)} clusters")
    print("-" * 60)

    for _, row in multi_item_clusters.head(top_k).iterrows():
        items = row["items"]
        # Show first 10 items, then "..." if more
        items_preview = items[:10]
        items_str = ", ".join(str(i) for i in items_preview)
        if len(items) > 10:
            items_str += f", ... (+{len(items) - 10} more)"
        print(f"Cluster {row['cluster_id']:3d} (size {row['size']:4d}): {items_str}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

