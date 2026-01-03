#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from gw2ml.data.retriever import PROJECT_ROOT, get_data_path
from gw2ml.data.s3_sync import get_s3_client


def load_env_file(env_path: Path) -> None:
    """Populate os.environ with values defined in the given .env file."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if not key:
            continue

        os.environ.setdefault(key, value)


def parse_extensions(raw_extensions: str | None) -> tuple[str, ...]:
    if not raw_extensions:
        return (".csv",)

    cleaned = []
    for entry in raw_extensions.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        cleaned.append(candidate if candidate.startswith(".") else f".{candidate}")
    return tuple({ext.lower() for ext in cleaned}) or (".csv",)


def resolve_data_root(explicit_data_root: str | None) -> Path:
    return get_data_path(base_path=explicit_data_root, ensure_exists=False)


def resolve_folder_path(folder_argument: str, data_root: Path) -> Path:
    candidate = Path(folder_argument).expanduser()
    if candidate.is_absolute():
        return candidate
    return (data_root / candidate).resolve()


def collect_files(folder_path: Path, extensions: Sequence[str]) -> list[Path]:
    matched_files: list[Path] = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and (not extensions or file_path.suffix.lower() in extensions):
            matched_files.append(file_path)
    return sorted(matched_files)


def ensure_trailing_slash(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def upload_files(
    *,
    files: Sequence[Path],
    base_folder: Path,
    bucket_name: str,
    prefix: str,
    workers: int,
    verbose: bool,
) -> tuple[int, int]:
    s3_client = get_s3_client()
    safe_workers = max(1, min(workers, len(files)))
    uploaded = 0
    failed = 0

    def _upload_single(file_path: Path, relative_key: str) -> None:
        s3_key = f"{prefix}{relative_key}"
        s3_client.upload_file(str(file_path), bucket_name, s3_key)

    with ThreadPoolExecutor(max_workers=safe_workers) as executor:
        future_to_key = {}
        for file_path in files:
            relative_key = file_path.relative_to(base_folder).as_posix()
            future = executor.submit(_upload_single, file_path, relative_key)
            future_to_key[future] = relative_key

        for future in as_completed(future_to_key):
            relative_key = future_to_key[future]
            try:
                future.result()
                uploaded += 1
                prefix_hint = f" → {prefix}{relative_key}" if verbose else ""
                print(f"✓ {relative_key}{prefix_hint}")
            except Exception as error:  # noqa: BLE001
                failed += 1
                print(f"✗ {relative_key} - {error}")

    return uploaded, failed


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a local data folder to Hetzner S3 with optional concurrency",
    )
    parser.add_argument(
        "folder",
        help="Relative (to DATA_PATH) or absolute path of the folder to upload.",
    )
    parser.add_argument(
        "--bucket",
        default="ost-s3",
        help="S3 bucket name (default: ost-s3)",
    )
    parser.add_argument(
        "--prefix",
        help="Override the S3 folder prefix. Defaults to datasources/gw2/raw/<timestamp>/",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        help="Optional Unix timestamp to reuse for the destination prefix.",
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        help="Override DATA_PATH resolution (supports relative paths from repo root).",
    )
    parser.add_argument(
        "--filter-ext",
        default=".csv",
        help="Comma-separated list of file extensions to upload (default: .csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent upload workers (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching files without uploading anything.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file to load before resolving DATA_PATH (default: ./.env)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print destination keys for every uploaded file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = configure_parser()
    args = parser.parse_args(argv)

    env_path = (PROJECT_ROOT / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    load_env_file(env_path)

    data_root = resolve_data_root(args.data_root)
    folder_path = resolve_folder_path(args.folder, data_root)

    if not folder_path.exists():
        print(f"Error: folder {folder_path} does not exist", file=sys.stderr)
        return 1

    extensions = parse_extensions(args.filter_ext)
    files = collect_files(folder_path, extensions)

    if not files:
        print("No files matched the requested extensions. Nothing to upload.")
        return 0

    timestamp = args.timestamp or int(time.time())
    prefix = ensure_trailing_slash(
        args.prefix or f"datasources/gw2/raw/{timestamp}/",
    )

    print(f"Resolved data root: {data_root}")
    print(f"Uploading from: {folder_path}")
    print(f"Total files: {len(files)} (filtered by {', '.join(extensions)})")
    print(f"Bucket: {args.bucket}")
    print(f"Prefix: {prefix}")
    print(f"Timestamp: {timestamp}")
    print()

    if args.dry_run:
        print("Dry run enabled. The following files would be uploaded:")
        for file_path in files:
            print(f"- {file_path.relative_to(folder_path).as_posix()}")
        return 0

    uploaded, failed = upload_files(
        files=files,
        base_folder=folder_path,
        bucket_name=args.bucket,
        prefix=prefix,
        workers=args.workers,
        verbose=args.verbose,
    )

    print("\n" + "=" * 50)
    print("Upload Summary:")
    print(f"  Uploaded: {uploaded}/{len(files)}")
    print(f"  Failed:   {failed}/{len(files)}")
    print(f"  Bucket:   {args.bucket}")
    print(f"  Prefix:   {prefix}")
    print(f"  Timestamp: {timestamp} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))})")
    print("=" * 50)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


