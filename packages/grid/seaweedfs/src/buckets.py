# stdlib
from enum import Enum
from enum import unique
import json
from pathlib import Path
from typing import Any

# third party
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

__all__ = [
    "BucketType",
    "BaseBucket",
    "AzureCreds",
    "AzureBucket",
    "GCSCreds",
    "GCSBucket",
    "S3Creds",
    "S3Bucket",
]


@unique
class BucketType(Enum):
    """Weed shell supported values"""

    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"


class BaseBucket(BaseModel):
    type: BucketType
    bucket_name: str

    @property
    def full_bucket_name(self) -> str:
        raise NotImplementedError()


def check_creds(v: Any) -> Any:
    if isinstance(v, Path) and not v.exists():
        raise FileNotFoundError(f"Credentials file not found: {v}")
    return v


def check_and_read_creds(v: Any) -> Any:
    """Check if creds are provided as a path to a JSON file, load them if so."""

    v = check_creds(v)
    if isinstance(v, str | Path):
        return json.loads(Path(v).read_text())

    # if not sure, just return the value as is
    return v


# ---------------------------------------------------------------------------------------------
# Azure Buckets
#
# Azure credentials are provided either as a dict (for API) or JSON file path (for Automount)
# While mounting, Seaweed expects the credentials to be provided as raw values
# Hence we don't need to maintain reference to the file, just read it and pass the values
# ---------------------------------------------------------------------------------------------


class AzureCreds(BaseModel):
    azure_account_name: str = Field(min_length=1)
    azure_account_key: str = Field(min_length=1)


class AzureBucket(BaseBucket):
    type: BucketType = BucketType.AZURE
    bucket_name: str = Field(alias="container_name", min_length=1)
    creds: AzureCreds

    @field_validator("creds", mode="before")
    @classmethod
    def validate_creds(cls, v: Any) -> Any:
        return check_and_read_creds(v)

    @property
    def full_bucket_name(self) -> str:
        return f"{self.creds.azure_account_name}/{self.bucket_name}"


# ---------------------------------------------------------------------------------------------
# S3 Buckets
#
# S3 credentials are provided either as a dict (for API) or JSON file path (for Automount)
# While mounting, Seaweed expects the credentials to be provided as raw values
# Hence we don't need to maintain reference to the file, just read it and pass the values
# ---------------------------------------------------------------------------------------------


class S3Creds(BaseModel):
    aws_access_key_id: str = Field(min_length=1)
    aws_secret_access_key: str = Field(min_length=1)


class S3Bucket(BaseBucket):
    type: BucketType = BucketType.S3
    bucket_name: str = Field(min_length=1)
    creds: S3Creds

    @field_validator("creds", mode="before")
    @classmethod
    def validate_creds(cls, v: Any) -> Any:
        return check_and_read_creds(v)

    @property
    def full_bucket_name(self) -> str:
        return self.bucket_name


# ---------------------------------------------------------------------------------------------
# GCS Buckets
#
# GCS credentials are provided either as a dict (for API) or JSON file path (for Automount)
# While mounting, Seaweed expects the credentials to be provided as a path to the JSON file
# ---------------------------------------------------------------------------------------------


class GCSCreds(BaseModel):
    val: Path | dict

    def read(self) -> dict:
        if isinstance(self.val, Path):
            return json.loads(self.val.read_text())
        return self.val

    def save(self, default: Path | None = None) -> Path:
        # if the value is already a path, just return it
        if isinstance(self.val, Path):
            return self.val
        elif default is None:
            raise ValueError("No path provided to save GCS credentials")

        # or save the value to the provided path
        default.write_text(json.dumps(self.val))
        return default


class GCSBucket(BaseBucket):
    type: BucketType = BucketType.GCS
    bucket_name: str = Field(min_length=1)
    creds: GCSCreds

    @field_validator("creds", mode="before")
    @classmethod
    def validate_creds(cls, v: Any) -> GCSCreds:
        if isinstance(v, Path) and not v.exists():
            raise FileNotFoundError(f"GCS credentials file not found: {v}")

        # we don't need to read the file, as we can just pass the path to Seaweed
        return GCSCreds(val=v)

    @property
    def full_bucket_name(self) -> str:
        project_id = self.creds.read().get("project_id", "")
        return f"{project_id}/{self.bucket_name}"
