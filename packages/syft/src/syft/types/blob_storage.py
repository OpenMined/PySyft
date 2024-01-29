# stdlib
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
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from azure.storage.blob import BlobSasPermissions
from azure.storage.blob import generate_blob_sas
from botocore.client import ClientError as BotoClientError
from typing_extensions import Self

# relative
from ..node.credentials import SyftVerifyKey
from ..serde import serialize
from ..serde.serializable import serializable
from ..service.action.action_object import ActionObject
from ..service.action.action_object import BASE_PASSTHROUGH_ATTRS
from ..service.action.action_types import action_types
from ..service.response import SyftError
from ..service.response import SyftException
from ..service.service import from_api_or_context
from ..types.grid_url import GridURL
from ..types.transforms import drop
from ..types.transforms import keep
from ..types.transforms import make_set_default
from ..types.transforms import transform
from .datetime import DateTime
from .syft_migration import migrate
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SYFT_OBJECT_VERSION_2
from .syft_object import SYFT_OBJECT_VERSION_3
from .syft_object import SyftObject
from .uid import UID

READ_EXPIRATION_TIME = 1800  # seconds
DEFAULT_CHUNK_SIZE = 10000 * 1024


@serializable()
class BlobFileV1(SyftObject):
    __canonical_name__ = "BlobFile"
    __version__ = SYFT_OBJECT_VERSION_1

    file_name: str

    __repr_attrs__ = ["id", "file_name"]


class BlobFileV2(SyftObject):
    __canonical_name__ = "BlobFile"
    __version__ = SYFT_OBJECT_VERSION_2

    file_name: str
    syft_blob_storage_entry_id: Optional[UID] = None
    file_size: Optional[int] = None

    __repr_attrs__ = ["id", "file_name"]


@serializable()
class BlobFile(SyftObject):
    __canonical_name__ = "BlobFile"
    __version__ = SYFT_OBJECT_VERSION_3

    file_name: str
    syft_blob_storage_entry_id: Optional[UID] = None
    file_size: Optional[int] = None
    path: Optional[Path]
    uploaded = False

    __repr_attrs__ = ["id", "file_name"]

    def read(self, stream=False, chunk_size=DEFAULT_CHUNK_SIZE, force=False):
        # get blob retrieval object from api + syft_blob_storage_entry_id
        read_method = from_api_or_context(
            "blob_storage.read", self.syft_node_location, self.syft_client_verify_key
        )
        blob_retrieval_object = read_method(self.syft_blob_storage_entry_id)
        return blob_retrieval_object._read_data(
            stream=stream, chunk_size=chunk_size, _deserialize=False
        )

    @classmethod
    def upload_from_path(self, path, client):
        # syft absolute
        import syft as sy

        return sy.ActionObject.from_path(path=path).send(client).syft_action_data

    def _upload_to_blobstorage_from_api(self, api):
        if self.path is None:
            raise ValueError("cannot upload BlobFile, no path specified")
        storage_entry = CreateBlobStorageEntry.from_path(self.path)

        blob_deposit_object = api.services.blob_storage.allocate(storage_entry)

        if isinstance(blob_deposit_object, SyftError):
            return blob_deposit_object

        with open(self.path, "rb") as f:
            result = blob_deposit_object.write(f)

        if isinstance(result, SyftError):
            return result

        self.syft_blob_storage_entry_id = blob_deposit_object.blob_storage_entry_id
        self.uploaded = True

    def upload_to_blobstorage(self, client):
        self.syft_node_location = client.id
        self.syft_client_verify_key = client.verify_key
        return self._upload_to_blobstorage_from_api(client.api)

    def _iter_lines(self, chunk_size=DEFAULT_CHUNK_SIZE):
        """Synchronous version of the async iter_lines. This implementation
        is also optimized in terms of splitting chunks, making it faster for
        larger lines"""
        pending = None
        for chunk in self.read(stream=True, chunk_size=chunk_size):
            if b"\n" in chunk:
                if pending is not None:
                    chunk = pending + chunk
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

    def read_queue(self, queue, chunk_size, progress=False, buffer_lines=10000):
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

    def iter_lines(self, chunk_size=DEFAULT_CHUNK_SIZE, progress=False):
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

    def _coll_repr_(self):
        return {"file_name": self.file_name}


@migrate(BlobFile, BlobFileV1)
def downgrade_blobfile_v2_to_v1():
    return [
        drop(["syft_blob_storage_entry_id", "file_size"]),
    ]


@migrate(BlobFileV1, BlobFile)
def upgrade_blobfile_v1_to_v2():
    return [
        make_set_default("syft_blob_storage_entry_id", None),
        make_set_default("file_size", None),
    ]


class BlobFileType(type):
    pass


class BlobFileObjectPointer:
    pass


@serializable()
class BlobFileObject(ActionObject):
    __canonical_name__ = "BlobFileOBject"
    __version__ = SYFT_OBJECT_VERSION_1

    syft_internal_type: ClassVar[Type[Any]] = BlobFile
    syft_pointer_type = BlobFileObjectPointer
    syft_passthrough_attrs = BASE_PASSTHROUGH_ATTRS


@serializable()
class SecureFilePathLocation(SyftObject):
    __canonical_name__ = "SecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    path: str

    def __repr__(self) -> str:
        return f"{self.path}"

    def generate_url(self, *args):
        raise NotImplementedError


@serializable()
class SeaweedSecureFilePathLocationV1(SecureFilePathLocation):
    __canonical_name__ = "SeaweedSecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1

    upload_id: str


@serializable()
class SeaweedSecureFilePathLocation(SecureFilePathLocation):
    __canonical_name__ = "SeaweedSecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_2

    upload_id: Optional[str] = None

    def generate_url(self, connection, type_, bucket_name):
        try:
            url = connection.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket_name, "Key": self.path},
                ExpiresIn=READ_EXPIRATION_TIME,
            )

            # relative
            from ..store.blob_storage import BlobRetrievalByURL

            return BlobRetrievalByURL(
                url=GridURL.from_url(url), file_name=Path(self.path).name, type_=type_
            )
        except BotoClientError as e:
            raise SyftException(e)


@migrate(SeaweedSecureFilePathLocationV1, SeaweedSecureFilePathLocation)
def upgrade_seaweedsecurefilepathlocation_v1_to_v2():
    return [make_set_default("bucket_name", "")]


@migrate(SeaweedSecureFilePathLocation, SeaweedSecureFilePathLocationV1)
def downgrade_seaweedsecurefilepathlocation_v2_to_v1():
    return [
        drop(["bucket_name"]),
    ]


@serializable()
class AzureSecureFilePathLocation(SecureFilePathLocation):
    __canonical_name__ = "AzureSecureFilePathLocation"
    __version__ = SYFT_OBJECT_VERSION_1

    # upload_id: str
    azure_profile_name: str  # Used by Seaweedfs to refer to a remote config
    bucket_name: str

    def generate_url(self, connection, type_, *args):
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
class BlobStorageEntryV1(SyftObject):
    __canonical_name__ = "BlobStorageEntry"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    location: Union[SecureFilePathLocation, SeaweedSecureFilePathLocation]
    type_: Optional[Type]
    mimetype: str = "bytes"
    file_size: int
    uploaded_by: SyftVerifyKey
    created_at: DateTime = DateTime.now()

    __attr_searchable__ = ["bucket_name"]


@serializable()
class BlobStorageEntry(SyftObject):
    __canonical_name__ = "BlobStorageEntry"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    location: Union[SecureFilePathLocation, SeaweedSecureFilePathLocation]
    type_: Optional[Type]
    mimetype: str = "bytes"
    file_size: int
    no_lines: Optional[int] = 0
    uploaded_by: SyftVerifyKey
    created_at: DateTime = DateTime.now()
    bucket_name: Optional[str]

    __attr_searchable__ = ["bucket_name"]


@migrate(BlobStorageEntry, BlobStorageEntryV1)
def downgrade_blobstorageentry_v2_to_v1():
    return [
        drop(["no_lines", "bucket_name"]),
    ]


@migrate(BlobStorageEntryV1, BlobStorageEntry)
def upgrade_blobstorageentry_v1_to_v2():
    return [make_set_default("no_lines", 1), make_set_default("bucket_name", None)]


@serializable()
class BlobStorageMetadataV1(SyftObject):
    __canonical_name__ = "BlobStorageMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    type_: Optional[Type[SyftObject]]
    mimetype: str = "bytes"
    file_size: int


@serializable()
class BlobStorageMetadata(SyftObject):
    __canonical_name__ = "BlobStorageMetadata"
    __version__ = SYFT_OBJECT_VERSION_2

    type_: Optional[Type[SyftObject]]
    mimetype: str = "bytes"
    file_size: int
    no_lines: Optional[int] = 0


@migrate(BlobStorageMetadata, BlobStorageMetadataV1)
def downgrade_blobmeta_v2_to_v1():
    return [
        drop(["no_lines"]),
    ]


@migrate(BlobStorageMetadataV1, BlobStorageMetadata)
def upgrade_blobmeta_v1_to_v2():
    return [make_set_default("no_lines", 1)]


@serializable()
class CreateBlobStorageEntry(SyftObject):
    __canonical_name__ = "CreateBlobStorageEntry"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    type_: Optional[Type]
    mimetype: str = "bytes"
    file_size: int
    extensions: List[str] = []

    @classmethod
    def from_obj(cls, obj: SyftObject) -> Self:
        file_size = sys.getsizeof(serialize._serialize(obj=obj, to_bytes=True))
        return cls(file_size=file_size, type_=type(obj))

    @classmethod
    def from_path(cls, fp: Union[str, Path], mimetype: Optional[str] = None) -> Self:
        path = Path(fp)
        if not path.exists():
            raise SyftException(f"{fp} does not exist.")
        if not path.is_file():
            raise SyftException(f"{fp} is not a file.")

        if fp.suffix.lower() == ".jsonl":
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
def storage_entry_to_metadata():
    return [keep(["id", "type_", "mimetype", "file_size"])]


action_types[BlobFile] = BlobFileObject
