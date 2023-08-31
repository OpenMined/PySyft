"""Blob file storage

Contains blob file storage interfaces. See `on_disk.py` for an example of a concrete implementation.

BlobStorageClient, BlobStorageClientConfig and BlobStorageConnection
-----------------------------------------------------

```
blob_storage_client_cls: Type[BlobStorageClient]
blob_storage_client_config: BlobStorageClientConfig

blob_storage_client = blob_storage_client_cls(config=blob_storage_client_config)
```

BlobStorageClient implements context manager (`__enter__()` and `__exit__()`) to create a BlobStorageConnection.
`BlobStorageClient.__enter__()` creates a connection (BlobStorageConnection) to the file system (e.g. SeaweedFS).
BlobStorageConnection implements operations on files (read, write, ...).

```
with blob_storage_client as conn:
    conn.read(...)
```

See `blob_storage/service.py` for usage example.

Write/persist SyftObject to blob storage
----------------------------------------

- create a CreateBlobStorageEntry from SyftObject `create_blob_storage_entry = CreateBlobStorageEntry.from(obj)`
- pre-allocate the file object `blob_deposit = api.services.blob_storage.allocate(create_blob_storage_entry)`
  (this returns a BlobDeposit)
- use `BlobDeposit.write` to upload/save/persist the SyftObject
  `blob_deposit.write(sy.serialize(user_object, to_bytes=True))`

Read/retrieve SyftObject from blob storage
------------------------------------------

- get a BlobRetrieval from the id of the BlobStorageEntry of the SyftObject
  `blob_retrieval = api.services.blob_storage.read(blob_storage_entry_id)`
- use `BlobRetrieval.read` to retrieve the SyftObject `syft_object = blob_retrieval.read()`
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
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SecureFilePathLocation
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID


@serializable()
class BlobRetrieval(SyftObject):
    __canonical_name__ = "BlobRetrieval"
    __version__ = SYFT_OBJECT_VERSION_1

    def read(self) -> SyftObject:
        pass


@serializable()
class SyftObjectRetrieval(BlobRetrieval):
    __canonical_name__ = "SyftObjectRetrieval"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_object: bytes

    def read(self) -> SyftObject:
        return deserialize(self.syft_object, from_bytes=True)


@serializable()
class BlobRetrievalByURL(BlobRetrieval):
    __canonical_name__ = "BlobRetrievalByURL"
    __version__ = SYFT_OBJECT_VERSION_1

    url: str

    def read(self) -> SyftObject:
        pass


@serializable()
class BlobDeposit(SyftObject):
    __canonical_name__ = "BlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_1

    blob_storage_entry_id: UID

    def write(self, data: bytes) -> Union[SyftSuccess, SyftError]:
        pass


@serializable()
class BlobStorageClientConfig(BaseModel):
    pass


class BlobStorageConnection:
    def read(self, fp: SecureFilePathLocation) -> BlobRetrieval:
        raise NotImplementedError

    def allocate(self, obj: CreateBlobStorageEntry) -> SecureFilePathLocation:
        raise NotImplementedError

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        raise NotImplementedError


@serializable()
class BlobStorageClient(SyftBaseModel):
    config: BlobStorageClientConfig

    def __enter__(self) -> BlobStorageConnection:
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        raise NotImplementedError


class BlobStorageConfig(SyftBaseModel):
    client_type: Type[BlobStorageClient]
    client_config: BlobStorageClientConfig
