# stdlib
from typing import Union

# relative
from . import BlobStorageClient
from . import BlobStorageClientConfig
from . import BlobStorageConfig
from . import BlobStorageConnection
from . import SyftWriteResource
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.syft_object import SYFT_OBJECT_VERSION_1


@serializable()
class SeaweedFSSyftWriteResource(SyftWriteResource):
    __canonical_name__ = "SeaweedFSSyftWriteResource"
    __version__ = SYFT_OBJECT_VERSION_1

    def write(self, data: bytes) -> Union[SyftSuccess, SyftError]:
        pass


@serializable()
class SeaweedFSClientConfig(BlobStorageClientConfig):
    pass


@serializable()
class SeaweedFSClient(BlobStorageClient):
    config: SeaweedFSClientConfig

    def __enter__(self) -> BlobStorageConnection:
        pass

    def __exit__(self, *exc) -> None:
        pass


class SeaweedFSConfig(BlobStorageConfig):
    blob_storage_client = SeaweedFSClient
    blob_storage_client_config: SeaweedFSClientConfig
