# stdlib
from io import BytesIO
import logging
from typing import Annotated
from typing import Any
from typing import Literal
from typing import TypeVar
from typing import get_args

# third party
from pydantic import AnyUrl
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import UrlConstraints
from pydantic import field_validator
from pydantic import model_validator

# first party
import fsspec

# relative
from . import BlobRetrieval
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SecureFilePathLocation
from .base_storage import BlobStorage
from .base_storage import BlobStorageClientConfig
from .errors import BlobStorageAllocationError
from .errors import BlobStorageClientError
from .errors import BlobStorageNotFoundError
from .errors import BlobStorageReadError
from .errors import BlobStorageWriteError

logger = logging.getLogger("syft.blob_storage")

T = TypeVar("T")

# You can find all valid schemes via fsspec.available_protocols()
# We support the following for now:
SupportedProtocols = Literal["local", "gs", "s3", "az"]
valid_schemes: tuple[SupportedProtocols, ...] = get_args(SupportedProtocols)
StorageURL = Annotated[
    AnyUrl, UrlConstraints(host_required=True, allowed_schemes=list(valid_schemes))
]

DEFAULT_BUCKET_NAME = "syft"


class BaseFileSystemConfig(BaseModel):
    anon: bool = False


class GoogleCloudStorageConfig(BaseFileSystemConfig):
    type: Literal["gs"] = Field(exclude=True)
    token: str | dict
    project: str = None
    default_region: str = None


class S3StorageConfig(BaseFileSystemConfig):
    type: Literal["s3"] = Field(exclude=True)
    key: str = Field(alias="aws_access_key_id")
    secret: str = Field(alias="aws_secret_access_key")
    token: str = Field(None, alias="aws_session_token")
    endpoint_url: str = None
    client_kwargs: dict = None
    config_kwargs: dict = None
    s3_additional_kwargs: dict = None


class AzureStorageConfig(BaseFileSystemConfig):
    type: Literal["az"] = Field(exclude=True)
    account_key: str
    account_name: str = Field(alias="container_name")
    connection_string: str = None
    sas_token: str = None
    client_id: str = None
    client_secret: str = None
    tenant_id: str = None


class LocalStorageConfig(BaseModel):
    type: Literal["local"] = Field(exclude=True)
    auto_mkdir: bool = True


type StorageOptions = GoogleCloudStorageConfig | S3StorageConfig | AzureStorageConfig | LocalStorageConfig


class BlobStorageFilesystemConfig(BlobStorageClientConfig):
    storage_url: StorageURL
    storage_options: StorageOptions = Field(discriminator="type")
    prefix: str = ""
    bucket_name: str = None
    protocol: SupportedProtocols = None

    @model_validator(mode="after")
    def extract_protocol_and_bucket_name(self) -> "BlobStorageFilesystemConfig":
        self.protocol = self.storage_url.scheme
        self.bucket_name = self.storage_url.host
        assert self.protocol == self.storage_options.type, (
            f"storage_options.type ({self.storage_options.type}) diverges from the "
            f"storage_url protocol ({self.protocol})"
        )
        return self

    @staticmethod
    def extract_protocol(storage_url: str) -> SupportedProtocols:
        return StorageURL(storage_url).scheme


class BlobStorageFilesystem(BlobStorage):
    config: BlobStorageFilesystemConfig
    fs: fsspec.AbstractFileSystem = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def create_filesystem(self) -> "BlobStorageFilesystem":
        try:
            self.fs = fsspec.filesystem(
                self.config.protocol,
                **self.config.storage_options.dict(exclude_none=True),
            )
        except Exception as e:
            logger.debug(
                f"BlobStorage(init): failed to create filesystem ({self.config.protocol})",
                exc_info=e,
            )
            raise BlobStorageClientError(e)
        try:
            if self.config.bucket_name:
                self.fs.ls(self.config.bucket_name)
        except Exception as e:
            logger.debug(
                f"BlobStorage(init): unable to find bucket {self.config.bucket_name}",
                exc_info=e,
            )
            raise BlobStorageNotFoundError(e)
        return self

    def __enter__(self) -> "BlobStorageFilesystem":
        return self

    @property
    def storage_type(self):
        return self.fs.fsid

    def connect(self) -> "BlobStorageFilesystem":
        return self

    def get_blob_path(self, file_name: str) -> str:
        return (
            f"{self.config.storage_url}/{file_name}"
            if not self.config.prefix
            else f"{self.config.storage_url}/{self.config.prefix}/{file_name}"
        )

    def allocate(self, obj: CreateBlobStorageEntry) -> SecureFilePathLocation:
        try:
            return SecureFilePathLocation(path=self.get_blob_path(obj.file_name))
        except Exception as e:
            logger.debug(f'BlobStorage(allocate): failed to "allocate"', exc_info=e)
            raise BlobStorageAllocationError(e)

    def read(self, fp: SecureFilePathLocation, type_: type[T] | None) -> BlobRetrieval:
        print("blob_storage_fs: read")
        print("blob_storage_fs: fp", fp)
        print("blob_storage_fs: type_", type_)
        try:
            with self.fs.open(fp.path, mode="rb") as f:
                return BlobRetrieval(data=f.read(), type_=type_)
        except Exception as e:
            logger.debug(f"BlobStorage(read): failed to read", exc_info=e)
            raise BlobStorageReadError(e)

    def write(self, fp: SecureFilePathLocation, data: BytesIO) -> int:
        try:
            with self.fs.open(fp.path, mode="wb") as f:
                return f.write(data.read())
        except Exception as e:
            logger.debug(f"BlobStorage(write): failed to write", exc_info=e)
            raise BlobStorageWriteError(e)

    def delete(self, fp: SecureFilePathLocation) -> bool:
        try:
            self.fs.rm(fp.path)
            return True
        except Exception as e:
            logger.debug(f"BlobStorage(delete): failed to delete {fp.path}", exc_info=e)
            return False
