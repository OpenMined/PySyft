# stdlib
from enum import Enum
from typing import Type
from typing import Union

# relative
from ..types.syft_object import SyftObject
from ..types.uid import UID


class SyftFile:  # (location of the file)
    filename: str
    file_size: int
    path: str


class SecureFilePathLocation:
    file_id: UID

    # def read():
    #     raise NotImplemented

    # def write():
    #     raise NotImplemented


class FileClientConfig:
    pass


class OnDiskClientConfig(FileClientConfig):
    pass


class SeaweedClientConfig(FileClientConfig):
    pass


class FileClient:
    _config: FileClientConfig

    def __init__(self, config: FileClientConfig):
        pass

    def read(self, fp: SecureFilePathLocation):
        raise NotImplemented

    def write(self, fp: SecureFilePathLocation):
        raise NotImplemented


class OnDiskFileClient(FileClient):
    _config: OnDiskClientConfig

    def read(fp: SecureFilePathLocation):
        pass

    def write(fp: SecureFilePathLocation):
        pass


class OnDiskFilePathLocation(FileClient):
    _config: SeaweedClientConfig

    def read(fp: SecureFilePathLocation):
        pass

    def write(fp: SecureFilePathLocation):
        pass


class SeaweedFSClient:
    pass


class SeaweedFSFilePathLocation(SecureFilePathLocation):
    _client: SeaweedFSClient

    def read():
        pass

    def write():
        pass


class ProxyActionObject:
    id: UID
    location: SecureFilePathLocation
    type_: Type[SyftObject]
    dtype: Type
    size: tuple


class ProxyActionObjectWithClient(ProxyActionObject):
    file_system_type: Type[FileClient]
    file_system_config: FileClientConfig

    def read(self):
        with self.file_system_type(self.file_system_config) as client:
            return client.read(self.location)


def add_file_system_type(context):
    # context.output["FileSystemType"] = context.node.file_system_type

    context.output["file_system_type"] = context.node.file_system_type
    context.output["file_system_config"] = context.node.file_system_config
    return context


proxy_obj = proxy_action_obj.to(ProxyActionObjectWithClient)

obj = proxy_obj.read()

# Database -> SyftFile
# Database -> ProxyActionObject
# class SeaweedFSFile(SyftFile):
#     pass


# ProxyActionObject -> ProxyActionObjectWithClient -> ActionObject

class ActionObject:

    @property
    def syft_action_data(self):
        proxy_obj = ProxyActionObject -> ProxyActionObjectWithClient
        return proxy_obj.read()

    @syft_action_data.setter
    def syft_action_data(self, data):
        self.proxy_obj.write(data)
