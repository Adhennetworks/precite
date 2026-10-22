"""Client S3 unique (Cloudflare R2, MinIO, AWS, ...)."""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from precite.config import get_settings
from precite.utils.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_s3_client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )


def ensure_bucket(bucket: str | None = None) -> None:
    s = get_settings()
    bucket = bucket or s.s3_bucket
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            log.info("s3.bucket.create", bucket=bucket)
            client.create_bucket(Bucket=bucket)
        else:
            raise
