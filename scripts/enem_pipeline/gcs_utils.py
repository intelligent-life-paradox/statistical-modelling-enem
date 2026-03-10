from __future__ import annotations

from pathlib import Path
from google.cloud import storage


def upload_file(bucket_name: str, source_path: Path, destination_blob: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(str(source_path))
    return f"gs://{bucket_name}/{destination_blob}"


def download_file(bucket_name: str, source_blob: str, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(str(destination_path))
    return destination_path


def maybe_gcs_uri(bucket_name: str, blob_name: str) -> str:
    return f"gs://{bucket_name}/{blob_name}"