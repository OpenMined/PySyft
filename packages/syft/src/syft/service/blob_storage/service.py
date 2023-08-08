# stdlib
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.blob_storage import BlobDeposit
from ...store.blob_storage import BlobRetrieval
from ...store.document_store import DocumentStore
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from .stash import BlobStorageStash


@serializable()
class BlobStorageService(AbstractService):
    store: DocumentStore
    stash: BlobStorageStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = BlobStorageStash(store=store)

    @service_method(path="blob_storage.get_all", name="get_all")
    def get_all_blob_storage_entries(
        self, context: AuthedServiceContext
    ) -> Union[List[BlobStorageEntry], SyftError]:
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="blob_storage.get_by_uid", name="get_by_uid")
    def get_blob_storage_entry_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[BlobStorageEntry, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="blob_storage.read", name="read")
    def read(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[BlobRetrieval, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            with context.node.blob_storage_client as conn:
                return conn.read(result.ok().location)
        return SyftError(message=result.err())

    @service_method(path="blob_storage.allocate", name="allocate")
    def allocate(
        self, context: AuthedServiceContext, obj: CreateBlobStorageEntry
    ) -> Union[BlobDeposit, SyftError]:
        with context.node.blob_storage_client as conn:
            secure_location = conn.allocate(obj)

            blob_storage_entry = BlobStorageEntry(
                location=secure_location,
                type_=obj.type_,
                mimetype=obj.mimetype,
                file_size=obj.file_size,
                uploaded_by=context.credentials,
            )
            blob_deposit = conn.write(blob_storage_entry)

        result = self.stash.set(context.credentials, blob_storage_entry)
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        return blob_deposit

    @service_method(path="blob_storage.write_to_disk", name="write_to_disk")
    def write_to_disk(
        self, context: AuthedServiceContext, uid: UID, data: bytes
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(
            credentials=context.credentials,
            uid=uid,
        )
        if result.is_err():
            return SyftError(message=f"{result.err()}")

        obj: Optional[BlobStorageEntry] = result.ok()

        if obj is None:
            return SyftError(message=f"No blob storage entry exists for uid: {uid}")

        try:
            Path(obj.location.path).write_bytes(data)
            return SyftSuccess(message="File successfully saved.")
        except Exception as e:
            return SyftError(message=f"Failed to write object to disk: {e}")


TYPE_TO_SERVICE[BlobStorageEntry] = BlobStorageEntry
