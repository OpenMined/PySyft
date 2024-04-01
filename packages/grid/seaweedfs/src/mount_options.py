# stdlib

# stdlib
from enum import StrEnum
from pathlib import Path

# third party
from pydantic import BaseModel
from pydantic import Field


class BucketType(StrEnum):
    """Weed shell supported values"""

    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"


class BaseBucket(BaseModel):
    type: BucketType
    # credentials
    creds: Path | dict


class AzureBucket(BaseBucket):
    type: BucketType = BucketType.AZURE
    bucket_name: str = Field(alias="container_name")


class S3Bucket(BaseBucket):
    type: BucketType = BucketType.S3
    bucket_name: str
    region: str
    storage_class: str = "STANDARD"


class GCSBucket(BaseBucket):
    type: BucketType = BucketType.GCS
    bucket_name: str
    project_id: str | None = None


class MountOptions(BaseModel):
    local_bucket: str  # bucket to mount to
    remote_bucket: S3Bucket | GCSBucket | AzureBucket
