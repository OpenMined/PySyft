# stdlib
from typing import Union

# relative
from . import FileClient
from . import FileClientConfig
from . import FileClientConnection
from . import FileStoreConfig
from . import SyftWriteResource
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.syft_object import SYFT_OBJECT_VERSION_1


@serializable()
class SeaweedSyftWriteResource(SyftWriteResource):
    __canonical_name__ = "SeaweedSyftWriteResource"
    __version__ = SYFT_OBJECT_VERSION_1

    def write(self, data: bytes) -> Union[SyftSuccess, SyftError]:
        pass


@serializable()
class SeaweedClientConfig(FileClientConfig):
    pass


@serializable()
class SeaweedFSClient(FileClient):
    config: SeaweedClientConfig

    def __enter__(self) -> FileClientConnection:
        pass

    def __exit__(self, *exc) -> None:
        pass


class SeaweedFileStoreConfig(FileStoreConfig):
    file_client = SeaweedFSClient
    file_client_config: SeaweedClientConfig
