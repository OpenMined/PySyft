# stdlib

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .user_code import UserCodeStatusCollection


@instrument
@serializable()
class StatusStash(BaseUIDStoreStash):
    object_type = UserCodeStatusCollection
    settings: PartitionSettings = PartitionSettings(
        name=UserCodeStatusCollection.__canonical_name__,
        object_type=UserCodeStatusCollection,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store)
        self.store = store
        self.settings = self.settings
        self._object_type = self.object_type

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[UserCodeStatusCollection, str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return self.query_one(credentials=credentials, qks=qks)


@instrument
@serializable()
class UserCodeStatusService(AbstractService):
    store: DocumentStore
    stash: StatusStash

    def __init__(self, store: DocumentStore):
        self.store = store
        self.stash = StatusStash(store=store)

    @service_method(path="code_status.create", name="create", roles=ADMIN_ROLE_LEVEL)
    def create(
        self,
        context: AuthedServiceContext,
        status: UserCodeStatusCollection,
    ) -> UserCodeStatusCollection | SyftError:
        result = self.stash.set(
            credentials=context.credentials,
            obj=status,
        )
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(
        path="code_status.get_by_uid", name="get_by_uid", roles=GUEST_ROLE_LEVEL
    )
    def get_status(
        self, context: AuthedServiceContext, uid: UID
    ) -> UserCodeStatusCollection | SyftError:
        """Get the status of a user code item"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="code_status.get_all", name="get_all", roles=ADMIN_ROLE_LEVEL)
    def get_all(
        self, context: AuthedServiceContext
    ) -> list[UserCodeStatusCollection] | SyftError:
        """Get all user code item statuses"""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="code_status.remove", name="remove", roles=ADMIN_ROLE_LEVEL)
    def remove(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        """Remove a user code item status"""
        result = self.stash.delete_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())


TYPE_TO_SERVICE[UserCodeStatusCollection] = UserCodeStatusService
