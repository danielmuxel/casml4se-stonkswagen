# %%
from __future__ import annotations

import boto3
import os
import time
from pathlib import Path
from datetime import datetime
import hashlib
from typing import Union

def get_s3_client():
    """Initializes and returns a boto3 S3 client from environment variables."""
    hetzner_endpoint = os.getenv('HETZNER_S3_ENDPOINT')
    access_key = os.getenv('HETZNER_S3_ACCESS_KEY')
    secret_key = os.getenv('HETZNER_S3_SECRET_KEY')
    return boto3.client(
        's3',
        endpoint_url=hetzner_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='fsn1'  # or 'nbg1' depending on your region
    )

def calculate_etag(file_path):
    """Calculate ETag (MD5 hash) of a local file to match S3's ETag"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Upload all files from the raw folder
def upload_folder_to_s3(local_folder , s3_folder_prefix,  unix_timestamp, bucket_name="ost-s3"):
    """
    Upload all files from a local folder to Hetzner Object Storage
    """
    s3_client = get_s3_client()
    local_path = Path(local_folder)

    if not local_path.exists():
        print(f"Error: Folder {local_folder} does not exist")
        return

    # Get all CSV files
    files = list(local_path.glob('*.csv'))

    print(f"Found {len(files)} files to upload")
    print(f"Unix timestamp: {unix_timestamp}")
    print(f"Target: {bucket_name}/{s3_folder_prefix}")
    print()

    uploaded = 0
    failed = 0

    for file_path in files:
        # Create S3 key (path in bucket)
        s3_key = s3_folder_prefix + file_path.name

        try:
            # Upload file
            s3_client.upload_file(
                str(file_path),
                bucket_name,
                s3_key
            )
            uploaded += 1
            print(f"✓ {file_path.name}")
        except Exception as e:
            failed += 1
            print(f"✗ {file_path.name} - Error: {str(e)}")

    print(f"\n{'=' * 50}")
    print(f"Upload Summary:")
    print(f"  Uploaded: {uploaded}/{len(files)}")
    print(f"  Failed: {failed}/{len(files)}")
    print(f"  Location: {bucket_name}/{s3_folder_prefix}")
    print(f"  Timestamp: {unix_timestamp} ({datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'=' * 50}")

# Download all files from a folder in Hetzner Object Storage
def download_folder_from_s3(
    s3_folder_prefix: str,
    local_folder: Union[str, Path],
    bucket_name: str = "ost-s3",
    skip_existing: bool = True,
) -> Path:
    """
    Download all files from a folder in Hetzner Object Storage to a local folder

    Args:
        s3_folder_prefix: The S3 folder path (e.g., 'datasources/gw2/raw/1762686861/')
        local_folder: Local directory to save files to
        bucket_name: S3 bucket name (default: 'ost-s3')
        skip_existing: Skip files that already exist with matching ETag (default: True)
    """
    s3_client = get_s3_client()
    local_path = Path(local_folder).expanduser()

    # Create local folder if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Source: {bucket_name}/{s3_folder_prefix}")
    print(f"Target: {local_folder}")
    print()

    downloaded = 0
    skipped = 0
    failed = 0

    try:
        # List all objects in the S3 folder
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=s3_folder_prefix
        )

        if 'Contents' not in response:
            print(f"No files found in {bucket_name}/{s3_folder_prefix}")
            return

        files = response['Contents']
        print(f"Found {len(files)} files to download")
        print()

        for obj in files:
            s3_key = obj['Key']

            # Skip if it's just a folder marker
            if s3_key.endswith('/'):
                continue

            # Extract filename from S3 key
            filename = s3_key.split('/')[-1]
            local_file_path = local_path / filename

            # Check if file exists and ETag matches
            if skip_existing and local_file_path.exists():
                s3_etag = obj['ETag'].strip('"')
                local_etag = calculate_etag(local_file_path)

                if local_etag == s3_etag:
                    skipped += 1
                    print(f"⊘ {filename} (already exists, ETag matches)")
                    continue

            try:
                # Download file
                s3_client.download_file(
                    bucket_name,
                    s3_key,
                    str(local_file_path)
                )
                downloaded += 1
                print(f"✓ {filename}")
            except Exception as e:
                failed += 1
                print(f"✗ {filename} - Error: {str(e)}")

        print(f"\n{'=' * 50}")
        print(f"Download Summary:")
        print(f"  Downloaded: {downloaded}/{len(files)}")
        print(f"  Skipped: {skipped}/{len(files)}")
        print(f"  Failed: {failed}/{len(files)}")
        print(f"  Location: {local_path}")
        print(f"  Source: {bucket_name}/{s3_folder_prefix}")
        print(f"{'=' * 50}")

    except Exception as e:
        print(f"Error listing objects: {str(e)}")
        raise

    return local_path


# Example usage:
# download_folder_from_s3(
#     s3_folder_prefix='datasources/gw2/raw/1762686861/',
#     local_folder='gw2/downloaded',
#     bucket_name='ost-s3'
# )
# Upload the folder
#upload_folder_to_s3(local_folder, bucket_name, s3_folder_prefix)