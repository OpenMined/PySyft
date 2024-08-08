# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..job.job_sql_stash import ObjectStash
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .status_sql import UserCodeStatusCollectionDB
from .user_code import UserCodeStatusCollection


@instrument
@serializable(canonical_name="StatusStashSQL", version=1)
class StatusStashSQL(ObjectStash[UserCodeStatusCollection, UserCodeStatusCollectionDB]):
    object_type = UserCodeStatusCollection
    settings: PartitionSettings = PartitionSettings(
        name=UserCodeStatusCollection.__canonical_name__,
        object_type=UserCodeStatusCollection,
    )

    def __init__(self, store) -> None:
        super().__init__(store)


@instrument
@serializable(canonical_name="UserCodeStatusService", version=1)
class UserCodeStatusService(AbstractService):
    store: DocumentStore
    stash: StatusStashSQL

    def __init__(self, store: DocumentStore):
        self.store = store
        self.stash = StatusStashSQL(store=store)

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
