# stdlib
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from typing import Type
from typing import Union

# third party
from pydantic import PrivateAttr

# relative
from . import BlobStorageClient
from . import BlobStorageClientConfig
from . import BlobStorageConfig
from . import BlobStorageConnection
from . import SyftObjectResource
from . import SyftResource
from . import SyftWriteResource
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.file_object import CreateFileObject
from ...types.file_object import FileObject
from ...types.file_object import SecureFilePathLocation
from ...types.syft_object import SYFT_OBJECT_VERSION_1


@serializable()
class OnDiskSyftWriteResource(SyftWriteResource):
    __canonical_name__ = "OnDiskSyftWriteResource"
    __version__ = SYFT_OBJECT_VERSION_1

    def write(self, data: bytes) -> Union[SyftSuccess, SyftError]:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.blob_storage.write_to_disk(data=data, obj=self.file_object)


class OnDiskBlobStorageConnection(BlobStorageConnection):
    _base_directory: Path

    def __init__(self, base_directory: Path) -> None:
        self._base_directory = base_directory

    def read(self, fp: SecureFilePathLocation) -> SyftResource:
        return SyftObjectResource(
            syft_object=(self._base_directory / fp.path).read_bytes()
        )

    def allocate(self, obj: CreateFileObject) -> SecureFilePathLocation:
        return SecureFilePathLocation(
            path=str((self._base_directory / str(obj.id)).absolute())
        )

    def create_resource(self, obj: FileObject) -> SyftWriteResource:
        return OnDiskSyftWriteResource(file_object=obj)


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
    blob_storage_client: Type[BlobStorageClient] = OnDiskBlobStorageClient
    blob_storage_client_config: OnDiskBlobStorageClientConfig = (
        OnDiskBlobStorageClientConfig()
    )
