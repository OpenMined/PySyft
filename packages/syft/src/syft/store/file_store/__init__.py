"""Blob file storage

Contains blob file storage interfaces. See `on_disk.py` for example of a concrete implementation.

Write/persist SyftObject to blob storage
----------------------------------------

- create a CreateFileObject from SyftObject `create_file_object = CreateFileObject.from(obj)`
- pre-allocate the file object `write_resource = api.services.file.allocate(create_file_object)`
  (this returns a SyftWriteResource)
- use `SyftWriteResource.write` to upload/save/persist the SyftObject
  `write_resource.write(sy.serialize(user_object, to_bytes=True))`

Read/retrieve SyftObject from blob storage
------------------------------------------

- get a SyftResource from the id of the FileObject of the SyftObject
  `resource = api.services.file.read(file_object_id)`
- use `SyftResource.read` to retrieve the SyftObject `syft_object = resouce.read()`
"""


# stdlib
from typing import Type
from typing import Union

# third party
from pydantic import BaseModel

# relative
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.base import SyftBaseModel
from ...types.file_object import CreateFileObject
from ...types.file_object import FileObject
from ...types.file_object import SecureFilePathLocation
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject


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
class FileClientConfig(BaseModel):
    pass


class FileClientConnection:
    def read(self, fp: SecureFilePathLocation) -> SyftResource:
        raise NotImplementedError

    def allocate(self, obj: CreateFileObject) -> SecureFilePathLocation:
        raise NotImplementedError

    def create_resource(self, obj: FileObject) -> SyftWriteResource:
        raise NotImplementedError


@serializable()
class FileClient(SyftBaseModel):
    config: FileClientConfig

    def __enter__(self) -> FileClientConnection:
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        raise NotImplementedError


class FileStoreConfig(SyftBaseModel):
    file_client: Type[FileClient]
    file_client_config: FileClientConfig
