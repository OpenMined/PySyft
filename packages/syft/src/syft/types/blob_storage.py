# stdlib
from enum import Enum
import mimetypes
from pathlib import Path
import sys
from typing import Optional
from typing import Type
from typing import Union

# third party
from result import Err
from result import Ok
from typing_extensions import Self

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.serializable import serializable
from ..service.response import SyftException
from .datetime import DateTime
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .uid import UID


@serializable()
class SecureFilePathLocation(SyftObject):
    __canonical_name__ = "SecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    path: str


@serializable()
class UploadStatus(Enum):
    PENDING = "Pending"
    DONE = "Done"
    FAILED = "Failed"


DEFAULT_EXPIRATION_TIME = 1800  # in seconds


@serializable()
class BlobStorageEntry(SyftObject):
    __canonical_name__ = "BlobStorageEntry"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    location: SecureFilePathLocation
    type_: Optional[Type[SyftObject]]
    mimetype: str = "bytes"
    file_size: int
    uploaded_by: SyftVerifyKey
    created_at: DateTime = DateTime.now()
    status: UploadStatus = UploadStatus.PENDING
    expires_in: int = DEFAULT_EXPIRATION_TIME

    def is_valid(self) -> Union[Ok, Err]:
        current_time = DateTime.now()
        if self.status in (UploadStatus.DONE, UploadStatus.FAILED):
            return Err(f"Object already exists for the given storage entry: {self.id}.")

        if (current_time - self.created_at) > self.expires_in:
            self.status = UploadStatus.FAILED
            return Err("Object Expired. Please request a new object")
        return Ok(True)


# TODO: Rethink if policy is required
# DefaultBlobUploadPolicy = {
#     "Content-Type": "*",
#     "Max-Size": 1024 * 1024 * 5,
#     "Min-Size": 0,
#     "Extensions": "*",
# }


@serializable()
class CreateBlobStorageEntry(SyftObject):
    __canonical_name__ = "CreateBlobStorageEntry"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    type_: Optional[Type[SyftObject]]
    mimetype: str = "bytes"
    file_size: int

    @classmethod
    def from_obj(cls, obj: SyftObject) -> Self:
        return cls(file_size=sys.getsizeof(obj), type_=type(obj))

    @classmethod
    def from_path(cls, fp: Union[str, Path], mimetype: Optional[str] = None) -> Self:
        path = Path(fp)
        if not path.exists():
            raise SyftException(f"{fp} does not exist.")
        if not path.is_file():
            raise SyftException(f"{fp} is not a file.")

        if mimetype is None:
            mime_types = mimetypes.guess_type(fp)
            if len(mime_types) > 0 and mime_types[0] is not None:
                mimetype = mime_types[0]
            else:
                raise SyftException(
                    "mimetype could not be identified.\n"
                    "Please specify mimetype manually `from_path(..., mimetype = ...)`."
                )

        return cls(mimetype=mimetype, file_size=path.stat().st_size)
