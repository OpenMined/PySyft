# stdlib
from pathlib import Path
from typing import List
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.blob_storage import BlobDeposit
from ...store.blob_storage import BlobRetrieval
from ...store.document_store import DocumentStore
from ...types.blob_storage import CreateFileObject
from ...types.blob_storage import FileObject
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
    def get_all_file_objects(
        self, context: AuthedServiceContext
    ) -> Union[List[FileObject], SyftError]:
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="blob_storage.get_by_uid", name="get_by_uid")
    def get_file_object_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[FileObject, SyftError]:
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
        self, context: AuthedServiceContext, obj: CreateFileObject
    ) -> Union[BlobDeposit, SyftError]:
        with context.node.blob_storage_client as conn:
            secure_location = conn.allocate(obj)

            file_object = FileObject(
                location=secure_location,
                type_=obj.type_,
                mimetype=obj.mimetype,
                file_size=obj.file_size,
                uploaded_by=context.credentials,
            )
            blob_deposit = conn.write(file_object)

        self.stash.set(context.credentials, file_object)
        return blob_deposit

    @service_method(path="blob_storage.write_to_disk", name="write_to_disk")
    def write_to_disk(
        self, context: AuthedServiceContext, obj: FileObject, data: bytes
    ) -> Union[SyftSuccess, SyftError]:
        try:
            Path(obj.location.path).write_bytes(data)
            return SyftSuccess(message="File successfully saved.")
        except Exception as e:
            return SyftError(message=f"Failed to write object to disk: {e}")


TYPE_TO_SERVICE[FileObject] = FileObject
