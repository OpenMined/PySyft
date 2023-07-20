# stdlib
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from typing import Type
from typing import Union

# third party
from pydantic import BaseModel
from pydantic import PrivateAttr

# relative
from ..serde.deserialize import _deserialize as deserialize
from ..serde.serializable import serializable
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.base import SyftBaseModel
from ..types.file_object import CreateFileObject
from ..types.file_object import FileObject
from ..types.file_object import SecureFilePathLocation
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject


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


@serializable()
class SyftWriteResource(SyftObject):
    __canonical_name__ = "SyftWriteResource"
    __version__ = SYFT_OBJECT_VERSION_1
    file_object: FileObject

    def write(self, data: bytes) -> Union[SyftSuccess, SyftError]:
        pass


@serializable()
class OnDiskSyftWriteResource(SyftWriteResource):
    __canonical_name__ = "OnDiskSyftWriteResource"
    __version__ = SYFT_OBJECT_VERSION_1

    def write(self, data: bytes) -> Union[SyftSuccess, SyftError]:
        # relative
        from ..client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.file.write_to_disk(data=data, obj=self.file_object)


@serializable()
class SeaweedSyftWriteResource(SyftWriteResource):
    __canonical_name__ = "SeaweedSyftWriteResource"
    __version__ = SYFT_OBJECT_VERSION_1

    def write(self, data: bytes) -> None:
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

    def allocate(self, obj: CreateFileObject) -> SecureFilePathLocation:
        raise NotImplementedError

    def create_resource(self, obj: FileObject) -> SyftWriteResource:
        raise NotImplementedError


# SyftObject write ->  syft server -> syftresourcewrite (write)

# def save(syftobject):
#     WriteSyftResource <- api.service.file.allocate(CreateFileObject)
#     WriteSyftResource.write(...)

# class OnDiskWriteSyftResource:
#     file_object_id: UID
#     def write(obj: SyftObject):
#         api.service.file.upload(obj, file_object)

# )
# SyftResource -> SyftServer -> data or link

# syftclient - SyftResource.read() return the data

# syftclient -> CreateFileObject -> SyftServer -> WriteSyftResource
# syftclient -> WriteSyftResource.write(data)


# WriteSyftResource.write(data)


class OnDiskFileClientConnection(FileClientConnection):
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


class FileClient(SyftBaseModel):
    config: FileClientConfig

    def __enter__(self) -> FileClientConnection:
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        raise NotImplementedError


class OnDiskFileClient(FileClient):
    config: OnDiskFileClientConfig
    _connection: OnDiskFileClientConnection = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._connection = OnDiskFileClientConnection(self.config.base_directory)

    def __enter__(self) -> FileClientConnection:
        return self._connection

    def __exit__(self, *exc) -> None:
        pass


class SeaweedFSClient(FileClient):
    config: SeaweedClientConfig

    def __enter__(self) -> FileClientConnection:
        pass

    def __exit__(self, *exc) -> None:
        pass


class FileStoreConfig(SyftBaseModel):
    file_client: Type[FileClient]
    file_client_config: FileClientConfig


class OnDiskFileStoreConfig(FileStoreConfig):
    file_client: Type[FileClient] = OnDiskFileClient
    file_client_config: OnDiskFileClientConfig = OnDiskFileClientConfig()


class SeaweedFileStoreConfig(FileStoreConfig):
    file_client = SeaweedFSClient
