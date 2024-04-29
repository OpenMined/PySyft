# stdlib
from typing import Any

# third party
from pydantic import BaseModel
from pydantic import field_validator

# relative
from .buckets import AzureBucket
from .buckets import GCSBucket
from .buckets import S3Bucket

__all__ = ["MountOptions"]


class MountOptions(BaseModel):
    local_bucket: str  # bucket to mount to
    remote_bucket: S3Bucket | GCSBucket | AzureBucket

    @field_validator("remote_bucket", mode="before")
    @classmethod
    def check_remote_bucket(cls, v: Any) -> BaseModel:
        if v["type"] == "gcs":
            return GCSBucket(**v)
        if v["type"] == "s3":
            return S3Bucket(**v)
        if v["type"] == "azure":
            return AzureBucket(**v)
        raise ValueError(f"Invalid remote bucket type: {v['type']}")
