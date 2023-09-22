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
from typing import Optional
from typing import Type
from typing import Union
from urllib.request import urlretrieve

# third party
from pydantic import BaseModel
import requests
from typing_extensions import Self

# relative
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.base import SyftBaseModel
from ...types.blob_storage import BlobFile
from ...types.blob_storage import BlobFileType
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SecureFilePathLocation
from ...types.grid_url import GridURL
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.constants import DEFAULT_TIMEOUT


@serializable()
class BlobRetrieval(SyftObject):
    __canonical_name__ = "BlobRetrieval"
    __version__ = SYFT_OBJECT_VERSION_1

    type_: Optional[Type]
    file_name: str

    def read(self) -> Union[SyftObject, SyftError]:
        pass


@serializable()
class SyftObjectRetrieval(BlobRetrieval):
    __canonical_name__ = "SyftObjectRetrieval"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_object: bytes

    def read(self) -> Union[SyftObject, SyftError]:
        if self.type_ is BlobFileType:
            with open(self.file_name, "wb") as fp:
                fp.write(self.syft_object)
            return BlobFile(file_name=self.file_name)
        return deserialize(self.syft_object, from_bytes=True)


@serializable()
class BlobRetrievalByURL(BlobRetrieval):
    __canonical_name__ = "BlobRetrievalByURL"
    __version__ = SYFT_OBJECT_VERSION_1

    url: GridURL

    def read(self) -> Union[SyftObject, SyftError]:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is not None:
            blob_url = api.connection.to_blob_route(self.url.url_path)
        else:
            blob_url = self.url
        try:
            if self.type_ is BlobFileType:
                urlretrieve(str(blob_url), filename=self.file_name)  # nosec
                return BlobFile(file_name=self.file_name)
            response = requests.get(str(blob_url), timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return deserialize(response.content, from_bytes=True)
        except requests.RequestException as e:
            return SyftError(message=f"Failed to retrieve with Error: {e}")


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
    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        raise NotImplementedError

    def read(self, fp: SecureFilePathLocation, type_: Optional[Type]) -> BlobRetrieval:
        raise NotImplementedError

    def allocate(
        self, obj: CreateBlobStorageEntry
    ) -> Union[SecureFilePathLocation, SyftError]:
        raise NotImplementedError

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        raise NotImplementedError

    def delete(self, fp: SecureFilePathLocation) -> bool:
        raise NotImplementedError


@serializable()
class BlobStorageClient(SyftBaseModel):
    config: BlobStorageClientConfig

    def connect(self) -> BlobStorageConnection:
        raise NotImplementedError


@serializable()
class BlobStorageConfig(SyftBaseModel):
    client_type: Type[BlobStorageClient]
    client_config: BlobStorageClientConfig
