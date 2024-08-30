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
from collections.abc import Callable
from collections.abc import Generator
from io import BytesIO
import logging
from typing import Any

# third party
from pydantic import BaseModel
import requests
from typing_extensions import Self

# relative
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from ...service.response import SyftSuccess
from ...types.base import SyftBaseModel
from ...types.blob_storage import BlobFile
from ...types.blob_storage import BlobFileType
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import DEFAULT_CHUNK_SIZE
from ...types.blob_storage import SecureFilePathLocation
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.server_url import ServerURL
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ...types.uid import UID

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 20


@serializable()
class BlobRetrieval(SyftObject):
    __canonical_name__ = "BlobRetrieval"
    __version__ = SYFT_OBJECT_VERSION_1

    type_: type | None = None
    file_name: str
    syft_blob_storage_entry_id: UID | None = None
    file_size: int | None = None


@serializable()
class SyftObjectRetrieval(BlobRetrieval):
    __canonical_name__ = "SyftObjectRetrieval"
    __version__ = SYFT_OBJECT_VERSION_1

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

    def read(self, _deserialize: bool = True) -> SyftObject:
        return self._read_data(_deserialize=_deserialize)


def syft_iter_content(
    blob_url: str | ServerURL,
    chunk_size: int,
    max_retries: int = MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
) -> Generator:
    """Custom iter content with smart retries (start from last byte read)"""
    current_byte = 0
    for attempt in range(max_retries):
        headers = {"Range": f"bytes={current_byte}-"}
        try:
            with requests.get(
                str(blob_url), stream=True, headers=headers, timeout=(timeout, timeout)
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(
                    chunk_size=chunk_size, decode_unicode=False
                ):
                    current_byte += len(chunk)
                    yield chunk
            return  # If successful, exit the function
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.debug(
                    f"Attempt {attempt}/{max_retries} failed: {e} at byte {current_byte}. Retrying..."
                )
            else:
                logger.error(f"Max retries reached - {e}")
                raise SyftException(public_message=f"Max retries reached - {e}")


@serializable()
class BlobRetrievalByURL(BlobRetrieval):
    __canonical_name__ = "BlobRetrievalByURL"
    __version__ = SYFT_OBJECT_VERSION_1

    url: ServerURL | str
    proxy_server_uid: UID | None = None

    def read(self) -> SyftObject:
        if self.type_ is BlobFileType:
            return BlobFile(
                file_name=self.file_name,
                syft_client_verify_key=self.syft_client_verify_key,
                syft_server_location=self.syft_server_location,
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

        api = self.get_api_wrapped()

        if api.is_ok() and api.unwrap().connection and isinstance(self.url, ServerURL):
            api = api.unwrap()
            if self.proxy_server_uid is None:
                blob_url = api.connection.to_blob_route(  # type: ignore [union-attr]
                    self.url.url_path, host=self.url.host_or_ip
                )
            else:
                blob_url = api.connection.stream_via(  # type: ignore [union-attr]
                    self.proxy_server_uid, self.url.url_path
                )
                stream = True
        else:
            blob_url = self.url

        try:
            is_blob_file = self.type_ is not None and issubclass(
                self.type_, BlobFileType
            )
            if is_blob_file and stream:
                return syft_iter_content(blob_url, chunk_size)

            response = requests.get(str(blob_url), stream=stream)  # nosec
            resp_content = response.content
            response.raise_for_status()

            return (
                resp_content
                if is_blob_file
                else deserialize(resp_content, from_bytes=True)
            )
        except requests.RequestException as e:
            raise SyftException(public_message=f"Failed to retrieve with error: {e}")


@serializable()
class BlobDeposit(SyftObject):
    __canonical_name__ = "BlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_1

    blob_storage_entry_id: UID

    @as_result(SyftException)
    def write(self, data: BytesIO) -> SyftSuccess:
        raise NotImplementedError


@serializable(canonical_name="BlobStorageClientConfig", version=1)
class BlobStorageClientConfig(BaseModel):
    pass


class BlobStorageConnection:
    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(self, *exc: Any) -> None:
        raise NotImplementedError

    def read(self, fp: SecureFilePathLocation, type_: type | None) -> BlobRetrieval:
        raise NotImplementedError

    def allocate(self, obj: CreateBlobStorageEntry) -> SecureFilePathLocation:
        raise NotImplementedError

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        raise NotImplementedError

    def delete(self, fp: SecureFilePathLocation) -> bool:
        raise NotImplementedError


@serializable(canonical_name="BlobStorageClient", version=1)
class BlobStorageClient(SyftBaseModel):
    config: BlobStorageClientConfig

    def connect(self) -> BlobStorageConnection:
        raise NotImplementedError


@serializable(canonical_name="BlobStorageConfig", version=1)
class BlobStorageConfig(SyftBaseModel):
    client_type: type[BlobStorageClient]
    client_config: BlobStorageClientConfig
    min_blob_size: int = 0  # in MB
