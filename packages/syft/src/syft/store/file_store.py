# stdlib
from pathlib import Path
from tempfile import gettempdir
from typing import Optional
from typing import Type

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


class FileClientConfig:
    pass


class OnDiskFileClientConfig(FileClientConfig):
    base_directory: Path

    def __init__(self, base_directory: Optional[Path]) -> None:
        self.base_directory = (
            Path(gettempdir()) if base_directory is None else base_directory
        )


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


class FileClient:
    _config: FileClientConfig

    def __init__(self, config: Optional[FileClientConfig]):
        pass

    def __enter__(self) -> FileClientConnection:
        raise NotImplementedError

    def __exit__(self) -> None:
        raise NotImplementedError


class OnDiskFileClient(FileClient):
    _config: OnDiskFileClientConfig

    def __init__(self, config: Optional[OnDiskFileClientConfig] = None):
        self._config = OnDiskFileClientConfig() if config is None else config
        self._connection = OnDiskFileClientConnection(self._config.base_directory)

    def __enter__(self) -> FileClientConnection:
        return self._connection

    def __exit__(self) -> None:
        pass


class SeaweedFSClient(FileClient):
    _config: SeaweedClientConfig

    def __init__(self, config: Optional[SeaweedClientConfig]):
        pass

    def __enter__(self) -> FileClientConnection:
        pass

    def __exit__(self) -> None:
        pass


class FileStoreConfig:
    file_client: Type[FileClient]
    file_client_config: Optional[FileClientConfig]


class OnDiskFileStoreConfig(FileStoreConfig):
    file_client = OnDiskFileClient
    file_client_config: Optional[FileClientConfig] = OnDiskFileClientConfig()


class SeaweedFileStoreConfig(FileStoreConfig):
    file_client = SeaweedFSClient
