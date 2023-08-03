# stdlib
from typing import Union

# relative
from . import BlobDeposit
from . import BlobStorageClient
from . import BlobStorageClientConfig
from . import BlobStorageConfig
from . import BlobStorageConnection
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.syft_object import SYFT_OBJECT_VERSION_1


@serializable()
class SeaweedFSBlobDeposit(BlobDeposit):
    __canonical_name__ = "SeaweedFSBlobDeposit"
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
    client_type = SeaweedFSClient
    client_config: SeaweedFSClientConfig
