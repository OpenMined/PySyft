# stdlib
from io import BytesIO
from pathlib import Path
from typing import Any

# third party
from typing_extensions import Self

from ...serde.serializable import serializable
from ...service.response import SyftError, SyftSuccess
from ...types.blob_storage import (
    BlobStorageEntry,
    CreateBlobStorageEntry,
    SecureFilePathLocation,
)
from ...types.syft_object import SYFT_OBJECT_VERSION_1

# relative
from . import (
    BlobDeposit,
    BlobRetrieval,
    BlobStorageClient,
    BlobStorageClientConfig,
    BlobStorageConfig,
    BlobStorageConnection,
    SyftObjectRetrieval,
)


@serializable()
class OnDiskBlobDeposit(BlobDeposit):
    __canonical_name__ = "OnDiskBlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_1

    def write(self, data: BytesIO) -> SyftSuccess | SyftError:
        # relative
        from ...service.service import from_api_or_context

        write_to_disk_method = from_api_or_context(
            func_or_path="blob_storage.write_to_disk",
            syft_server_location=self.syft_server_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        if write_to_disk_method is None:
            return SyftError(message="write_to_disk_method is None")
        return write_to_disk_method(data=data.read(), uid=self.blob_storage_entry_id)


class OnDiskBlobStorageConnection(BlobStorageConnection):
    _base_directory: Path

    def __init__(self, base_directory: Path) -> None:
        self._base_directory = base_directory

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def read(
        self, fp: SecureFilePathLocation, type_: type | None, **kwargs: Any,
    ) -> BlobRetrieval:
        file_path = self._base_directory / fp.path
        return SyftObjectRetrieval(
            syft_object=file_path.read_bytes(),
            file_name=file_path.name,
            type_=type_,
        )

    def allocate(
        self, obj: CreateBlobStorageEntry,
    ) -> SecureFilePathLocation | SyftError:
        try:
            return SecureFilePathLocation(
                path=str((self._base_directory / obj.file_name).absolute()),
            )
        except Exception as e:
            return SyftError(message=f"Failed to allocate: {e}")

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        return OnDiskBlobDeposit(blob_storage_entry_id=obj.id)

    def delete(self, fp: SecureFilePathLocation) -> SyftSuccess | SyftError:
        try:
            (self._base_directory / fp.path).unlink()
            return SyftSuccess(message="Successfully deleted file.")
        except FileNotFoundError as e:
            return SyftError(message=f"Failed to delete file: {e}")


@serializable(canonical_name="OnDiskBlobStorageClientConfig", version=1)
class OnDiskBlobStorageClientConfig(BlobStorageClientConfig):
    base_directory: Path


@serializable(canonical_name="OnDiskBlobStorageClient", version=1)
class OnDiskBlobStorageClient(BlobStorageClient):
    config: OnDiskBlobStorageClientConfig

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.config.base_directory.mkdir(exist_ok=True)

    def connect(self) -> BlobStorageConnection:
        return OnDiskBlobStorageConnection(self.config.base_directory)


@serializable(canonical_name="OnDiskBlobStorageConfig", version=1)
class OnDiskBlobStorageConfig(BlobStorageConfig):
    client_type: type[BlobStorageClient] = OnDiskBlobStorageClient
    client_config: OnDiskBlobStorageClientConfig
