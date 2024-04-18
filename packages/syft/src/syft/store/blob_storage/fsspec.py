# stdlib
import logging
from pathlib import Path
from typing import Annotated
from typing import Literal
from typing import TypeVar
from typing import get_args

# third party
import fsspec
from pydantic import AnyUrl
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import UrlConstraints
from pydantic import model_validator

# relative
from . import BlobRetrieval
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SecureFilePathLocation
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from .base_storage import BlobStorage
from .base_storage import BlobStorageConfig
from .errors import BlobStorageAllocationError
from .errors import BlobStorageClientError
from .errors import BlobStorageDeleteError
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
    anon: bool = False  # anonymous access to blob storage


class GoogleCloudStorageConfig(BaseFileSystemConfig):
    type: Literal["gs"] = Field(exclude=True)
    token: str | dict
    project: str | None = None
    default_region: str | None = None


class S3StorageConfig(BaseFileSystemConfig):
    type: Literal["s3"] = Field(exclude=True)
    key: str = Field(alias="aws_access_key_id")
    secret: str = Field(alias="aws_secret_access_key")
    token: str = Field(None, alias="aws_session_token")
    bucket_name: str | None = None
    endpoint_url: str | None = None
    client_kwargs: dict | None = None
    config_kwargs: dict | None = None
    s3_additional_kwargs: dict | None = None


class AzureStorageConfig(BaseFileSystemConfig):
    type: Literal["az"] = Field(exclude=True)
    account_key: str
    account_name: str = Field(alias="container_name")
    connection_string: str | None = None
    sas_token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    tenant_id: str | None = None


class LocalStorageConfig(BaseModel):
    type: Literal["local"] = Field(exclude=True)
    auto_mkdir: bool = True


StorageOptions = (
    GoogleCloudStorageConfig | S3StorageConfig | AzureStorageConfig | LocalStorageConfig
)


class BlobStorageFilesystemConfig(BlobStorageConfig):
    storage_url: StorageURL
    storage_options: StorageOptions = Field(discriminator="type")
    prefix: str | None = None
    bucket_name: str | None = None
    protocol: SupportedProtocols | None = None

    @model_validator(mode="after")
    def extract_protocol(self) -> "BlobStorageFilesystemConfig":
        self.protocol = self.storage_url.scheme
        assert self.protocol == self.storage_options.type, (
            f"storage_options.type ({self.storage_options.type}) diverges from the "
            f"storage_url protocol ({self.protocol})"
        )
        return self

    @model_validator(mode="after")
    def extract_bucket_name(self) -> "BlobStorageFilesystemConfig":
        match self.protocol:
            case "gs":
                self.bucket_name = self.storage_url.host
            case "s3":
                self.bucket_name = (
                    self.storage_options.bucket_name or DEFAULT_BUCKET_NAME
                )
            case "az":
                pass  # TODO: Azure bucket_name setup
            case "local":
                # TODO: get_temp_dir should be solved in node.py
                path = Path(self.storage_url.host)
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                self.bucket_name = str(path.absolute())
            case _:
                self.bucket_name = DEFAULT_BUCKET_NAME
        return self

    @staticmethod
    def get_protocol(storage_url: str) -> SupportedProtocols:
        return StorageURL(storage_url).scheme


@serializable()
class TempBlobRetrieval(BlobRetrieval):
    __canonical_name__ = "TempBlobRetrieval"
    __version__ = SYFT_OBJECT_VERSION_1

    data: bytes

    # TODO: Refactor, work this out...
    def _read_data(
        self,
        stream: bool = False,
        chunk_size: int = 10000 * 1024,
        deserialize: bool = True,
    ) -> list[bytes] | bytes:
        res = _deserialize(self.data, from_bytes=True) if deserialize else self.data
        # TODO: implement proper streaming from local files
        return [res] if stream else res

    def read(
        self, stream: bool = False, chunk_size: int = None, deserialize: bool = True
    ) -> list[bytes] | bytes:
        return self._read_data(
            stream=stream, chunk_size=chunk_size, deserialize=deserialize
        )


# TODO: Identify and raise permission denied errors
class BlobStorageFilesystem(BlobStorage):
    config: BlobStorageFilesystemConfig
    _fs: fsspec.AbstractFileSystem = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _create_filesystem(self) -> "BlobStorageFilesystem":
        try:
            self._fs = fsspec.filesystem(
                self.config.protocol,
                **self.config.storage_options.model_dump(exclude_none=True),
            )
        except Exception as e:
            logger.debug(
                f"BlobStorage(init): failed to create filesystem ({self.config.protocol})",
                exc_info=e,
            )
            raise BlobStorageClientError(e)
        try:  # Attempts to list the bucket to ensure it exists -- find a better way to test connection
            self.fs.ls(self.config.bucket_name)
        except Exception as e:
            logger.debug(
                f"BlobStorage(init): unable to find bucket {self.config.bucket_name}",
                exc_info=e,
            )
            raise BlobStorageNotFoundError(e)
        return self

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        return self._fs

    def storage_type(self) -> str:
        return self._fs.fsid

    def __enter__(self) -> "BlobStorageFilesystem":
        return self

    def connect(self) -> "BlobStorageFilesystem":
        return self

    def get_blob_path(self, file_name: str) -> str:
        return (
            f"{self.config.storage_url}/{file_name}"
            if not self.config.prefix
            else f"{self.config.storage_url}/{self.config.prefix}/{file_name}"
        )

    def allocate(self, obj: CreateBlobStorageEntry) -> SecureFilePathLocation:
        # TODO: Should allocate 'touch' objects? This is a glorified path builder
        #       see comments about fsspec.touch and Azure
        # TODO: Should we check if it is possible to put this file in the bucket?
        #       i.e. Check the if there's enough space to save it. Fail this early?...
        try:
            return SecureFilePathLocation(path=self.get_blob_path(obj.id))
        except Exception as e:
            logger.debug('BlobStorage(allocate): failed to "allocate"', exc_info=e)
            raise BlobStorageAllocationError(e)

    def read(self, fp: SecureFilePathLocation, type_: type[T] | None) -> BlobRetrieval:
        print("we called read in ", type(self))
        try:
            print(f"called read fsspec {fp.path}")
            with self.fs.open(fp.path, mode="rb") as f:
                read_data = f.read()
            return TempBlobRetrieval(data=read_data, type_=type_, file_name=fp.path)
        except Exception as e:
            logger.debug("BlobStorage(read): failed to read", exc_info=e)
            raise BlobStorageReadError(e)

    def write(self, fp: SecureFilePathLocation, data: bytes) -> int:
        try:
            with self.fs.open(fp.path, mode="wb") as f:  # Check if other modes needed
                return f.write(data)
        except Exception as e:
            logger.debug("BlobStorage(write): failed to write", exc_info=e)
            raise BlobStorageWriteError(e)

    def delete(self, fp: SecureFilePathLocation) -> True:
        try:
            self.fs.rm(fp.path)
            return True
        except Exception as e:
            logger.debug(f"BlobStorage(delete): failed to delete {fp.path}", exc_info=e)
            raise BlobStorageDeleteError(e)
