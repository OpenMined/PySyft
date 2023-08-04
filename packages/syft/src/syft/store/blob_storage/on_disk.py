# stdlib
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from typing import Type
from typing import Union

# third party
from pydantic import PrivateAttr

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

    def write(self, data: bytes) -> Union[SyftSuccess, SyftError]:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.blob_storage.write_to_disk(
            data=data, uid=self.blob_storage_entry_id
        )


class OnDiskBlobStorageConnection(BlobStorageConnection):
    _base_directory: Path

    def __init__(self, base_directory: Path) -> None:
        self._base_directory = base_directory

    def read(self, fp: SecureFilePathLocation) -> BlobRetrieval:
        return SyftObjectRetrieval(
            syft_object=(self._base_directory / fp.path).read_bytes()
        )

    def allocate(self, obj: CreateBlobStorageEntry) -> SecureFilePathLocation:
        return SecureFilePathLocation(
            path=str((self._base_directory / str(obj.id)).absolute())
        )

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        return OnDiskBlobDeposit(blob_storage_entry_id=obj.id)


@serializable()
class OnDiskBlobStorageClientConfig(BlobStorageClientConfig):
    base_directory: Path = Path(gettempdir())


@serializable()
class OnDiskBlobStorageClient(BlobStorageClient):
    config: OnDiskBlobStorageClientConfig
    _connection: OnDiskBlobStorageConnection = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._connection = OnDiskBlobStorageConnection(self.config.base_directory)

    def __enter__(self) -> BlobStorageConnection:
        return self._connection

    def __exit__(self, *exc) -> None:
        pass


class OnDiskBlobStorageConfig(BlobStorageConfig):
    client_type: Type[BlobStorageClient] = OnDiskBlobStorageClient
    client_config: OnDiskBlobStorageClientConfig = OnDiskBlobStorageClientConfig()
