# stdlib
from abc import ABC
from abc import abstractmethod
from io import BytesIO
from types import TracebackType
from typing import overload

# third party
from pydantic import BaseModel

# relative
from . import BlobRetrieval
from ...serde.serializable import serializable
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SecureFilePathLocation


@serializable()
class BlobStorageConfig(BaseModel):
    pass


class BlobStorage(BaseModel, ABC):
    config: BlobStorageConfig

    @abstractmethod
    def __enter__(self) -> "BlobStorage":
        raise NotImplementedError

    @overload
    def __exit__(self, exc_type: None, exc_value: None, traceback: None) -> None: ...

    @overload
    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    @abstractmethod
    def connect(self) -> "BlobStorage":
        raise NotImplementedError

    @abstractmethod
    def storage_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_blob_path(self, blob_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def read(self, fp: SecureFilePathLocation, type_: type | None) -> BlobRetrieval:
        raise NotImplementedError

    @abstractmethod
    def allocate(self, obj: CreateBlobStorageEntry) -> SecureFilePathLocation:
        raise NotImplementedError

    @abstractmethod
    def write(self, fp: SecureFilePathLocation, data: BytesIO) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete(self, fp: SecureFilePathLocation) -> bool:
        raise NotImplementedError
