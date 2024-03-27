# stdlib
from typing import Literal

# third party
from pydantic import BaseModel
from pydantic import Field


class AzureBucket(BaseModel):
    type: Literal["azure"] = "azure"
    bucket_name: str = Field(alias="container_name")
    # credentials
    azure_account_name: str
    azure_account_key: str


class S3Bucket(BaseModel):
    type: Literal["s3"] = "s3"
    bucket_name: str
    storage_class: str = "STANDARD"
    # credentials
    aws_access_key: str
    aws_secret_key_id: str


class GCSBucket(BaseModel):
    type: Literal["gcs"] = "gcs"
    bucket_name: str
    # credentials
    gcs_credentials: dict
