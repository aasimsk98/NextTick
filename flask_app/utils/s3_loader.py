"""
Download model artifacts from S3 to local cache dir at app startup.

EC2 uses IAM role for auth (no keys needed).
Local dev uses ~/.aws/credentials via `aws configure`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 7 artifacts in S3 bucket
MODEL_FILES = [
    "logistic_regression.pkl",
    "random_forest_classifier.pkl",
    "linear_regression.pkl",
    "random_forest_regressor.pkl",
    "lstm_classifier.pt",
    "lstm_regressor.pt",
    "scaler.pkl",
]


def download_models_from_s3(
    bucket: str,
    local_dir: str,
    prefix: str = "",
) -> str:
    """
    Download all model files from S3 bucket to local_dir.
    Skips files already present (cached across container restarts if vol mounted).

    Returns local_dir for chaining.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    for fname in MODEL_FILES:
        local_path = os.path.join(local_dir, fname)
        s3_key = f"{prefix}{fname}" if prefix else fname

        if os.path.exists(local_path):
            logger.info("Cached: %s (skip download)", fname)
            continue

        try:
            logger.info("Downloading s3://%s/%s -> %s", bucket, s3_key, local_path)
            s3.download_file(bucket, s3_key, local_path)
        except ClientError as exc:
            logger.error("S3 download failed for %s: %s", s3_key, exc)
            raise RuntimeError(
                f"Failed to download {s3_key} from s3://{bucket}. "
                f"Check IAM permissions + bucket name. ({exc})"
            )

    logger.info("All %d model files ready in %s", len(MODEL_FILES), local_dir)
    return local_dir