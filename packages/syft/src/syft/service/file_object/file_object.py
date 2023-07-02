# stdlib
from typing import Type

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.file_store import SecureFilePathLocation
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID


@serializable()
class FileObject(SyftObject):
    __canonical_name__ = "FileObject"
    __version__ = SYFT_OBJECT_VERSION_1
    id: UID
    location: SecureFilePathLocation
    type_: Type[SyftObject]
    mimetype: str = "bytes"
    file_size: int
    uploaded_by: SyftVerifyKey
    create_at: DateTime = DateTime.now()
