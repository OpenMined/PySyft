# stdlib
from collections.abc import Callable
from collections.abc import Iterator
from datetime import datetime
from datetime import timedelta
import mimetypes
from pathlib import Path
from queue import Queue
import sys
import threading
from time import sleep
from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING

# third party
from azure.storage.blob import BlobSasPermissions
from azure.storage.blob import generate_blob_sas
from botocore.client import ClientError as BotoClientError
from typing_extensions import Self

# relative
from ..client.api import SyftAPI
from ..client.client import SyftClient
from ..serde import serialize
from ..serde.serializable import serializable
from ..server.credentials import SyftVerifyKey
from ..service.action.action_object import ActionObject
from ..service.action.action_object import ActionObjectPointer
from ..service.action.action_object import BASE_PASSTHROUGH_ATTRS
from ..service.action.action_types import action_types
from ..service.service import from_api_or_context
from ..types.errors import SyftException
from ..types.server_url import ServerURL
from ..types.transforms import keep
from ..types.transforms import transform
from .datetime import DateTime
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .uid import UID

if TYPE_CHECKING:
    # relative
    from ..store.blob_storage import BlobRetrievalByURL
    from ..store.blob_storage import BlobStorageConnection


READ_EXPIRATION_TIME = 1800  # seconds
DEFAULT_CHUNK_SIZE = 10000 * 1024


@serializable()
class BlobFile(SyftObject):
    __canonical_name__ = "BlobFile"
    __version__ = SYFT_OBJECT_VERSION_1

    file_name: str
    syft_blob_storage_entry_id: UID | None = None
    file_size: int | None = None
    path: Path | None = None
    uploaded: bool = False

    __repr_attrs__ = ["id", "file_name"]

    def read(
        self,
        stream: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        force: bool = False,
    ) -> Any:
        # get blob retrieval object from api + syft_blob_storage_entry_id
        read_method = from_api_or_context(
            "blob_storage.read", self.syft_server_location, self.syft_client_verify_key
        )
        if read_method is not None:
            blob_retrieval_object = read_method(self.syft_blob_storage_entry_id)
            return blob_retrieval_object._read_data(
                stream=stream, chunk_size=chunk_size, _deserialize=False
            )
        else:
            return None

    @classmethod
    def upload_from_path(cls, path: str | Path, client: SyftClient) -> Any:
        # syft absolute
        import syft as sy

        return sy.ActionObject.from_path(path=path).send(client).syft_action_data

    def _upload_to_blobstorage_from_api(self, api: SyftAPI) -> None:
        if self.path is None:
            raise ValueError("cannot upload BlobFile, no path specified")
        storage_entry = CreateBlobStorageEntry.from_path(self.path)

        blob_deposit_object = api.services.blob_storage.allocate(storage_entry)

        with open(self.path, "rb") as f:
            blob_deposit_object.write(f).unwrap()

        self.syft_blob_storage_entry_id = blob_deposit_object.blob_storage_entry_id
        self.uploaded = True

        return None

    def upload_to_blobstorage(self, client: SyftClient) -> None:
        self.syft_server_location = client.id
        self.syft_client_verify_key = client.verify_key
        return self._upload_to_blobstorage_from_api(client.api)

    def _iter_lines(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
        """Synchronous version of the async iter_lines. This implementation
        is also optimized in terms of splitting chunks, making it faster for
        larger lines"""
        pending = None
        for chunk in self.read(stream=True, chunk_size=chunk_size):
            if b"\n" in chunk:
                if pending is not None:
                    chunk = pending + chunk  # type: ignore[unreachable]
                lines = chunk.splitlines()
                if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                    pending = lines.pop()
                else:
                    pending = None
                yield from lines
            else:
                if pending is None:
                    pending = chunk
                else:
                    pending = pending + chunk

        if pending is not None:
            yield pending

    def read_queue(
        self,
        queue: Queue,
        chunk_size: int,
        progress: bool = False,
        buffer_lines: int = 10000,
    ) -> None:
        total_read = 0
        for _i, line in enumerate(self._iter_lines(chunk_size=chunk_size)):
            line_size = len(line) + 1  # add byte for \n
            if self.file_size is not None:
                total_read = min(self.file_size, total_read + line_size)
            else:
                # naive way of doing this, max be 1 byte off because the last
                # byte can also be a \n
                total_read += line_size
            if progress:
                queue.put((total_read, line))
            else:
                queue.put(line)
            while queue.qsize() > buffer_lines:
                sleep(0.1)
        # Put anything not a string at the end
        queue.put(0)

    def iter_lines(
        self, chunk_size: int = DEFAULT_CHUNK_SIZE, progress: bool = False
    ) -> Iterator[str]:
        item_queue: Queue = Queue()
        threading.Thread(
            target=self.read_queue,
            args=(item_queue, chunk_size, progress),
            daemon=True,
        ).start()
        item = item_queue.get()
        while item != 0:
            yield item
            item = item_queue.get()

    def _coll_repr_(self) -> dict[str, str]:
        return {"file_name": self.file_name}


@serializable(canonical_name="BlobFileType", version=1)
class BlobFileType(type):
    pass


@serializable(canonical_name="BlobFileObjectPointer", version=1)
class BlobFileObjectPointer(ActionObjectPointer):
    pass


@serializable()
class BlobFileObject(ActionObject):
    __canonical_name__ = "BlobFileOBject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[type[Any]] = BlobFile
    syft_pointer_type: ClassVar[type[ActionObjectPointer]] = BlobFileObjectPointer
    syft_passthrough_attrs: list[str] = BASE_PASSTHROUGH_ATTRS


@serializable()
class SecureFilePathLocation(SyftObject):
    __canonical_name__ = "SecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    path: str

    def __repr__(self) -> str:
        return f"{self.path}"

    def generate_url(
        self,
        connection: "BlobStorageConnection",
        type_: type | None,
        bucket_name: str | None,
        *args: Any,
    ) -> "BlobRetrievalByURL":
        raise NotImplementedError


@serializable()
class SeaweedSecureFilePathLocation(SecureFilePathLocation):
    __canonical_name__ = "SeaweedSecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1

    upload_id: str | None = None

    def generate_url(
        self,
        connection: "BlobStorageConnection",
        type_: type | None,
        bucket_name: str | None,
        *args: Any,
    ) -> "BlobRetrievalByURL":
        try:
            url = connection.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket_name, "Key": self.path},
                ExpiresIn=READ_EXPIRATION_TIME,
            )

            # relative
            from ..store.blob_storage import BlobRetrievalByURL

            return BlobRetrievalByURL(
                url=ServerURL.from_url(url), file_name=Path(self.path).name, type_=type_
            )
        except BotoClientError as e:
            raise SyftException(e)


@serializable()
class AzureSecureFilePathLocation(SecureFilePathLocation):
    __canonical_name__ = "AzureSecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1

    # upload_id: str
    azure_profile_name: str  # Used by Seaweedfs to refer to a remote config
    bucket_name: str

    def generate_url(
        self, connection: "BlobStorageConnection", type_: type | None, *args: Any
    ) -> "BlobRetrievalByURL":
        # SAS is almost the same thing as the presigned url
        config = connection.config.remote_profiles[self.azure_profile_name]
        account_name = config.account_name
        container_name = config.container_name
        blob_name = self.path
        sas_blob = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=config.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=48),
        )
        url = f"https://{config.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_blob}"

        # relative
        from ..store.blob_storage import BlobRetrievalByURL

        return BlobRetrievalByURL(url=url, file_name=Path(self.path).name, type_=type_)


@serializable()
class BlobStorageEntry(SyftObject):
    __canonical_name__ = "BlobStorageEntry"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    location: SecureFilePathLocation | SeaweedSecureFilePathLocation
    type_: type | None = None
    mimetype: str = "bytes"
    file_size: int
    no_lines: int | None = 0
    uploaded_by: SyftVerifyKey
    created_at: DateTime = DateTime.now()
    bucket_name: str | None = None

    __attr_searchable__ = ["bucket_name"]


@serializable()
class BlobStorageMetadata(SyftObject):
    __canonical_name__ = "BlobStorageMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    type_: type[SyftObject] | None = None
    mimetype: str = "bytes"
    file_size: int
    no_lines: int | None = 0


@serializable()
class CreateBlobStorageEntry(SyftObject):
    __canonical_name__ = "CreateBlobStorageEntry"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    type_: type | None = None
    mimetype: str = "bytes"
    file_size: int
    extensions: list[str] = []

    @classmethod
    def from_blob_storage_entry(cls, entry: BlobStorageEntry) -> Self:
        # TODO extensions are not stored in the BlobStorageEntry,
        # so a blob entry from path might get a different filename
        # after uploading.
        return cls(
            id=entry.id,
            type_=entry.type_,
            mimetype=entry.mimetype,
            file_size=entry.file_size,
        )

    @classmethod
    def from_obj(cls, obj: SyftObject, file_size: int | None = None) -> Self:
        if file_size is None:
            file_size = sys.getsizeof(serialize._serialize(obj=obj, to_bytes=True))
        return cls(file_size=file_size, type_=type(obj))

    @classmethod
    def from_path(cls, fp: str | Path, mimetype: str | None = None) -> Self:
        path = Path(fp)
        if not path.exists():
            raise SyftException(f"{fp} does not exist.")
        if not path.is_file():
            raise SyftException(f"{fp} is not a file.")

        if path.suffix.lower() == ".jsonl":
            mimetype = "application/json-lines"
        if mimetype is None:
            mime_types = mimetypes.guess_type(fp)
            if len(mime_types) > 0 and mime_types[0] is not None:
                mimetype = mime_types[0]
            else:
                raise SyftException(
                    "mimetype could not be identified.\n"
                    "Please specify mimetype manually `from_path(..., mimetype = ...)`."
                )

        return cls(
            mimetype=mimetype,
            file_size=path.stat().st_size,
            extensions=path.suffixes,
            type_=BlobFileType,
        )

    @property
    def file_name(self) -> str:
        return str(self.id) + "".join(self.extensions)


@transform(BlobStorageEntry, BlobStorageMetadata)
def storage_entry_to_metadata() -> list[Callable]:
    return [keep(["id", "type_", "mimetype", "file_size"])]


action_types[BlobFile] = BlobFileObject
