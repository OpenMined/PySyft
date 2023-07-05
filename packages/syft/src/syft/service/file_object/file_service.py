# stdlib
from typing import List
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.file_store import SyftResource
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from .file_object import FileObject
from .file_stash import FileObjectStash


@serializable()
class FileObjectService(AbstractService):
    store: DocumentStore
    stash: FileObjectStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = FileObjectStash(store=store)

    @service_method(path="file.get_all", name="get_all")
    def get_all_file_objects(
        self, context: AuthedServiceContext
    ) -> Union[List[FileObject], SyftError]:
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="file.get_by_uid", name="get_by_uid")
    def get_file_object_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[FileObject, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="file.read", name="read")
    def read(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftResource, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            with context.node.file_client as conn:
                return conn.read(result.ok().location)
        return SyftError(message=result.err())


TYPE_TO_SERVICE[FileObject] = FileObject
