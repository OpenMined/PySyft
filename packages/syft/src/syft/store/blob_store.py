# stdlib
from pathlib import Path
from typing import List
from typing import Type

# relative
from ..serde.serializable import serializable
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.transforms import TransformContext
from ..types.transforms import transform
from ..types.uid import UID


@serializable()
class SecureFilePathLocation(SyftObject):
    __canonical_name__ = "SecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1
    id: UID
    path: str


class FileClientConfig:
    pass


class OnDiskClientConfig(FileClientConfig):
    pass


class SeaweedClientConfig(FileClientConfig):
    pass


class FileClientConnection:
    def read(self, fp: SecureFilePathLocation) -> bytes:
        raise NotImplemented

    def write(fp: SecureFilePathLocation, data: bytes) -> None:
        raise NotImplemented


class OnDiskFileClientConnection(FileClientConnection):
    def read(self, fp: SecureFilePathLocation) -> bytes:
        return Path(fp.path).read_bytes()

    def write(self, fp: SecureFilePathLocation, data: bytes) -> None:
        Path(fp.path).write_bytes(data)


class FileClient:
    _config: FileClientConfig

    def __init__(self, config: FileClientConfig):
        pass

    def __enter__(self) -> FileClientConnection:
        raise NotImplemented

    def __exit__(self) -> None:
        raise NotImplemented


class OnDiskFileClient(FileClient):
    _config: OnDiskClientConfig

    def __init__(self, config: OnDiskClientConfig = OnDiskClientConfig()):
        pass

    def __enter__(self) -> FileClientConnection:
        return OnDiskFileClientConnection()

    def __exit__(self) -> None:
        pass


class SeaweedFSClient(FileClient):
    _config: SeaweedClientConfig

    def __init__(self, config: SeaweedClientConfig):
        pass

    def __enter__(self) -> FileClientConnection:
        pass

    def __exit__(self) -> None:
        pass


class ProxyActionObject:
    id: UID
    location: SecureFilePathLocation
    type_: Type[SyftObject]
    dtype: Type
    size: tuple


class ProxyActionObjectWithClient(ProxyActionObject):
    file_client: FileClient

    def read(self):
        with self.file_client as fp:
            fp.read(self.location)


def add_file_client(context: TransformContext) -> TransformContext:
    context.output["file_client"] = context.node.file_system_type(context.node.file_system_config)
    return context


@transform(ProxyActionObject, ProxyActionObjectWithClient)
def attach_client() -> List[callable]:
    return [add_file_client]


class ActionObject:
    @property
    def syft_action_data(self):
        proxy_obj = ProxyActionObject -> ProxyActionObjectWithClient
        return proxy_obj.read()

    @syft_action_data.setter
    def syft_action_data(self, data):
        self.proxy_obj.write(data)
