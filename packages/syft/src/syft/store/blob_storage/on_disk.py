# stdlib
from io import BytesIO
from pathlib import Path
from typing import Any

# third party
from typing_extensions import Self

# relative
from . import BlobDeposit
from . import BlobRetrieval
from . import BlobStorageClient
from . import BlobStorageClientConfig
from . import BlobStorageConfig
from . import BlobStorageConnection
from . import SyftObjectRetrieval
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SecureFilePathLocation
from ...types.syft_object import SYFT_OBJECT_VERSION_1


@serializable()
class OnDiskBlobDeposit(BlobDeposit):
    __canonical_name__ = "OnDiskBlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_1

    def write(self, data: BytesIO) -> SyftSuccess | SyftError:
        # relative
        from ...service.service import from_api_or_context

        get_by_uid_method = from_api_or_context(
            func_or_path="blob_storage.get_by_uid",
            syft_server_location=self.syft_server_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        if get_by_uid_method is None:
            return SyftError(message="get_by_uid_method is None")

        obj = get_by_uid_method(uid=self.blob_storage_entry_id)
        if isinstance(obj, SyftError):
            return obj
        if obj is None:
            return SyftError(
                message=f"No blob storage entry exists for uid: {self.blob_storage_entry_id}, "
                "or you have no permissions to read it"
            )

        try:
            Path(obj.location.path).write_bytes(data.read())
            return SyftSuccess(message="File successfully saved.")
        except Exception as e:
            return SyftError(message=f"Failed to write object to disk: {e}")


class OnDiskBlobStorageConnection(BlobStorageConnection):
    _base_directory: Path

    def __init__(self, base_directory: Path) -> None:
        self._base_directory = base_directory

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def read(
        self, fp: SecureFilePathLocation, type_: type | None, **kwargs: Any
    ) -> BlobRetrieval:
        file_path = self._base_directory / fp.path
        return SyftObjectRetrieval(
            syft_object=file_path.read_bytes(),
            file_name=file_path.name,
            type_=type_,
        )

    def allocate(
        self, obj: CreateBlobStorageEntry
    ) -> SecureFilePathLocation | SyftError:
        try:
            return SecureFilePathLocation(
                path=str((self._base_directory / obj.file_name).absolute())
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
