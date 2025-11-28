import os
import boto3
from moto import mock_aws
from pathlib import Path
import pytest
import time

# Mock environment variables before importing the module
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("HETZNER_S3_ENDPOINT", "https://s3.amazonaws.com")
    monkeypatch.setenv("HETZNER_S3_ACCESS_KEY", "testing")
    monkeypatch.setenv("HETZNER_S3_SECRET_KEY", "testing")

# Now import the module
from gw2ml.data.s3_sync import upload_folder_to_s3, download_folder_from_s3

TEST_BUCKET = "stonkswagen-test-bucket"

@pytest.fixture
def s3_client():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=TEST_BUCKET)
        yield s3

def test_upload_and_download_s3_sync(s3_client, tmp_path):
    """
    Tests the full cycle of uploading a folder to S3, downloading it,
    and verifying the contents.
    """
    # 1. Setup: Create a source directory with dummy files
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    file1_content = "This is file one."
    file2_content = "This is file two."
    (source_dir / "file1.csv").write_text(file1_content)
    (source_dir / "file2.csv").write_text(file2_content)

    # 2. Test Upload
    s3_prefix = "test_upload/"
    timestamp = int(time.time())
    upload_folder_to_s3(
        local_folder=str(source_dir),
        s3_folder_prefix=s3_prefix,
        unix_timestamp=timestamp,
        bucket_name=TEST_BUCKET
    )

    # 3. Verify Upload
    response = s3_client.list_objects_v2(Bucket=TEST_BUCKET, Prefix=s3_prefix)
    assert 'Contents' in response
    uploaded_files = {obj['Key'] for obj in response['Contents']}
    assert f"{s3_prefix}file1.csv" in uploaded_files
    assert f"{s3_prefix}file2.csv" in uploaded_files

    # 4. Setup for Download: Create a destination directory
    dest_dir = tmp_path / "destination"
    dest_dir.mkdir()

    # 5. Test Download
    download_folder_from_s3(
        s3_folder_prefix=s3_prefix,
        local_folder=str(dest_dir),
        bucket_name=TEST_BUCKET
    )

    # 6. Verify Download
    downloaded_file1 = dest_dir / "file1.csv"
    downloaded_file2 = dest_dir / "file2.csv"
    assert downloaded_file1.exists()
    assert downloaded_file2.exists()
    assert downloaded_file1.read_text() == file1_content
    assert downloaded_file2.read_text() == file2_content

    # 7. Test skip_existing functionality
    # Modify a local file, so it should be re-downloaded
    (dest_dir / "file1.csv").write_text("modified content")

    # Download again, only one file should be downloaded
    downloaded = 0
    skipped = 0
    def new_print(*args, **kwargs):
        nonlocal downloaded, skipped
        if args:
            if "✓" in args[0]:
                downloaded += 1
            if "⊘" in args[0]:
                skipped += 1

    # Temporarily patch print to count downloads
    original_print = __builtins__["print"]
    __builtins__["print"] = new_print
    download_folder_from_s3(
        s3_folder_prefix=s3_prefix,
        local_folder=str(dest_dir),
        bucket_name=TEST_BUCKET,
        skip_existing=True
    )
    __builtins__["print"] = original_print

    # After modifying file1.csv, it should be re-downloaded.
    # file2.csv should be skipped.
    assert downloaded == 1
    assert skipped == 1
    assert (dest_dir / "file1.csv").read_text() == file1_content