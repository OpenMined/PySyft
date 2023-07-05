# stdlib
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from typing import Type

# third party
from pydantic import BaseModel

# relative
from ..serde.deserialize import _deserialize as deserialize
from ..serde.serializable import serializable
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID


@serializable()
class SecureFilePathLocation(SyftObject):
    __canonical_name__ = "SecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1
    id: UID
    path: str


@serializable()
class SyftResource(SyftObject):
    __canonical_name__ = "SyftResource"
    __version__ = SYFT_OBJECT_VERSION_1

    def read(self) -> SyftObject:
        pass


@serializable()
class SyftObjectResource(SyftObject):
    __canonical_name__ = "SyftObjectResource"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_object: bytes

    def read(self) -> SyftObject:
        return deserialize(self.syft_object, from_bytes=True)


@serializable()
class SyftURLResource(SyftObject):
    __canonical_name__ = "SyftURLResource"
    __version__ = SYFT_OBJECT_VERSION_1

    url: str

    def read(self) -> SyftObject:
        pass


class FileClientConfig(BaseModel):
    pass


class OnDiskFileClientConfig(FileClientConfig):
    base_directory: Path = Path(gettempdir())


class SeaweedClientConfig(FileClientConfig):
    pass


class FileClientConnection:
    def read(self, fp: SecureFilePathLocation) -> SyftResource:
        raise NotImplementedError

    def write(fp: SecureFilePathLocation, data: bytes) -> None:
        raise NotImplementedError


class OnDiskFileClientConnection(FileClientConnection):
    _base_directory: Path

    def __init__(self, base_directory: Path) -> None:
        self._base_directory = base_directory

    def read(self, fp: SecureFilePathLocation) -> SyftResource:
        return SyftObjectResource(
            syft_object=(self._base_directory / fp.path).read_bytes()
        )

    def write(self, fp: SecureFilePathLocation, data: bytes) -> None:
        (self._base_directory / fp.path).write_bytes(data)


class FileClient(BaseModel):
    config: FileClientConfig

    def __enter__(self) -> FileClientConnection:
        raise NotImplementedError

    def __exit__(self) -> None:
        raise NotImplementedError


class OnDiskFileClient(FileClient):
    config: OnDiskFileClientConfig
    connection: OnDiskFileClientConnection

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.connection = OnDiskFileClientConnection(self.config.base_directory)

    def __enter__(self) -> FileClientConnection:
        return self.connection

    def __exit__(self) -> None:
        pass


class SeaweedFSClient(FileClient):
    config: SeaweedClientConfig

    def __enter__(self) -> FileClientConnection:
        pass

    def __exit__(self) -> None:
        pass


class FileStoreConfig(BaseModel):
    file_client: Type[FileClient]
    file_client_config: FileClientConfig


class OnDiskFileStoreConfig(FileStoreConfig):
    file_client: Type[FileClient] = OnDiskFileClient
    file_client_config: OnDiskFileClientConfig = OnDiskFileClientConfig()


class SeaweedFileStoreConfig(FileStoreConfig):
    file_client = SeaweedFSClient
