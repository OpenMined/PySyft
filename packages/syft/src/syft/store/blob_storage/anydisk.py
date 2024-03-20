# stdlib
from io import BytesIO
from typing import Any

# third party
import smart_open
from typing_extensions import Self

# relative
from . import BlobDeposit
from . import BlobRetrieval
from . import BlobStorageClient
from . import BlobStorageClientConfig
from . import BlobStorageConfig
from . import BlobStorageConnection
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SecureFilePathLocation
from ...types.syft_object import SYFT_OBJECT_VERSION_1


def smart_read(path: str, **kwargs: Any) -> BytesIO:
    with smart_open.open(path, "rb", **kwargs) as fp:
        return BytesIO(fp.read())


def smart_write(path: str, data: BytesIO, **kwargs: Any) -> None:
    with smart_open.open(path, "wb", **kwargs) as fp:
        fp.write(data.read())


@serializable(without="smart_open_kwargs")
class AnyBlobStorageClientConfig(BlobStorageClientConfig):
    workspace: str
    bucket: str
    smart_open_kwargs: dict | None = None

    def get_blob_path(self, *args: str) -> str:
        _path = f"{self.bucket}/syft/{self.workspace}/blob/{'/'.join(args)}"
        print("AnyBlobStorageClientConfig.get_blob_path", _path)
        return _path


@serializable(without="config")
class AnyBlobDeposit(BlobDeposit):
    __canonical_name__ = "AnyBlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_1

    config: AnyBlobStorageClientConfig

    def write(self, data: BytesIO) -> SyftSuccess | SyftError:
        path = self.config.get_blob_path(str(self.blob_storage_entry_id))
        kwargs = self.config.smart_open_kwargs

        try:
            smart_write(path, data, **kwargs)
            return SyftSuccess(message="Successfully wrote file.")
        except Exception as e:
            return SyftError(message=f"Failed to write file: {e}")


class AnyBlobStorageConnection(BlobStorageConnection):
    _config: AnyBlobStorageClientConfig

    def __init__(self, config: AnyBlobStorageClientConfig) -> None:
        self._config = config

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def read(
        self, fp: SecureFilePathLocation, type_: type | None, **kwargs: Any
    ) -> BlobRetrieval:
        try:
            path = self._config.get_blob_path(fp.path)
            smart_kwargs = self._config.smart_open_kwargs
            data = smart_read(path, **smart_kwargs, **kwargs)
            return BlobRetrieval(data=data, type_=type_)
        except Exception as e:
            return SyftError(message=f"Failed to read file: {e}")

    def allocate(
        self, obj: CreateBlobStorageEntry
    ) -> SecureFilePathLocation | SyftError:
        try:
            path = self._config.get_blob_path(obj.file_name)
            return SecureFilePathLocation(path=path)
        except Exception as e:
            return SyftError(message=f"Failed to allocate: {e}")

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        return AnyBlobDeposit(blob_storage_entry_id=obj.id, config=self._config)

    def delete(self, fp: SecureFilePathLocation) -> SyftSuccess | SyftError:
        try:
            # TODO: smart_open does not support delete =)
            return SyftSuccess(message="Successfully deleted file.")
        except FileNotFoundError as e:
            return SyftError(message=f"Failed to delete file: {e}")


@serializable(without="config")
class AnyBlobStorageClient(BlobStorageClient):
    config: AnyBlobStorageClientConfig

    def __init__(self, **data: Any):
        super().__init__(**data)

    def connect(self) -> BlobStorageConnection:
        return AnyBlobStorageConnection(self.config)


@serializable()
class AnyBlobStorageConfig(BlobStorageConfig):
    client_type: type[BlobStorageClient] = AnyBlobStorageClient
    client_config: AnyBlobStorageClientConfig
