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
from collections.abc import Generator
from io import BytesIO
from typing import Any

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
from ...types.blob_storage import DEFAULT_CHUNK_SIZE
from ...types.blob_storage import SecureFilePathLocation
from ...types.grid_url import GridURL
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SYFT_OBJECT_VERSION_4
from ...types.syft_object import SyftObject
from ...types.uid import UID

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 20


@serializable()
class BlobRetrieval(SyftObject):
    __canonical_name__ = "BlobRetrieval"
    __version__ = SYFT_OBJECT_VERSION_3

    type_: type | None = None
    file_name: str
    syft_blob_storage_entry_id: UID | None = None
    file_size: int | None = None


@serializable()
class SyftObjectRetrieval(BlobRetrieval):
    __canonical_name__ = "SyftObjectRetrieval"
    __version__ = SYFT_OBJECT_VERSION_4

    syft_object: bytes

    def _read_data(
        self, stream: bool = False, _deserialize: bool = True, **kwargs: Any
    ) -> Any:
        # development setup, we can access the same filesystem
        if not _deserialize:
            res = self.syft_object
        else:
            res = deserialize(self.syft_object, from_bytes=True)

        # TODO: implement proper streaming from local files
        if stream:
            return [res]
        else:
            return res

    def read(self, _deserialize: bool = True) -> SyftObject | SyftError:
        return self._read_data(_deserialize=_deserialize)


def syft_iter_content(
    blob_url: str | GridURL,
    chunk_size: int,
    max_retries: int = MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
) -> Generator:
    """custom iter content with smart retries (start from last byte read)"""
    current_byte = 0
    for attempt in range(max_retries):
        try:
            headers = {"Range": f"bytes={current_byte}-"}
            with requests.get(
                str(blob_url), stream=True, headers=headers, timeout=(timeout, timeout)
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(
                    chunk_size=chunk_size, decode_unicode=False
                ):
                    current_byte += len(chunk)
                    yield chunk
                return

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(
                    f"Attempt {attempt}/{max_retries} failed: {e} at byte {current_byte}. Retrying..."
                )
            else:
                print(f"Max retries reached. Failed with error: {e}")
                raise


@serializable()
class BlobRetrievalByURL(BlobRetrieval):
    __canonical_name__ = "BlobRetrievalByURL"
    __version__ = SYFT_OBJECT_VERSION_4

    url: GridURL | str

    def read(self) -> SyftObject | SyftError:
        if self.type_ is BlobFileType:
            return BlobFile(
                file_name=self.file_name,
                syft_client_verify_key=self.syft_client_verify_key,
                syft_node_location=self.syft_node_location,
                syft_blob_storage_entry_id=self.syft_blob_storage_entry_id,
                file_size=self.file_size,
            )
        else:
            return self._read_data()

    def _read_data(
        self,
        stream: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api and api.connection and isinstance(self.url, GridURL):
            blob_url = api.connection.to_blob_route(
                self.url.url_path, host=self.url.host_or_ip
            )
        else:
            blob_url = self.url
        try:
            if self.type_ is BlobFileType:
                if stream:
                    return syft_iter_content(blob_url, chunk_size)
                else:
                    response = requests.get(str(blob_url), stream=False)  # nosec
                    response.raise_for_status()
                    return response.content
            else:
                response = requests.get(str(blob_url), stream=stream)  # nosec
                response.raise_for_status()
                return deserialize(response.content, from_bytes=True)
        except requests.RequestException as e:
            return SyftError(message=f"Failed to retrieve with Error: {e}")


@serializable()
class BlobDeposit(SyftObject):
    __canonical_name__ = "BlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_2

    blob_storage_entry_id: UID

    def write(self, data: BytesIO) -> SyftSuccess | SyftError:
        raise NotImplementedError


@serializable()
class BlobStorageClientConfig(BaseModel):
    pass


class BlobStorageConnection:
    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(self, *exc: Any) -> None:
        raise NotImplementedError

    def read(self, fp: SecureFilePathLocation, type_: type | None) -> BlobRetrieval:
        raise NotImplementedError

    def allocate(
        self, obj: CreateBlobStorageEntry
    ) -> SecureFilePathLocation | SyftError:
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
    client_type: type[BlobStorageClient]
    client_config: BlobStorageClientConfig
