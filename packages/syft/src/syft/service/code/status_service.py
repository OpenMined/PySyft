# stdlib

# third party

# relative
from ...client.api import ServerIdentity
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...store.db.stash import ObjectStash
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .user_code import ApprovalDecision
from .user_code import UserCodeStatusCollection


@serializable(canonical_name="StatusSQLStash", version=1)
class StatusStash(ObjectStash[UserCodeStatusCollection]):
    pass


class CodeStatusUpdate(PartialSyftObject):
    __canonical_name__ = "CodeStatusUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    decision: ApprovalDecision


@serializable(canonical_name="UserCodeStatusService", version=1)
class UserCodeStatusService(AbstractService):
    stash: StatusStash

    def __init__(self, store: DBManager):
        self.stash = StatusStash(store=store)

    @service_method(path="code_status.create", name="create", roles=ADMIN_ROLE_LEVEL)
    def create(
        self,
        context: AuthedServiceContext,
        status: UserCodeStatusCollection,
    ) -> UserCodeStatusCollection:
        res = self.stash.set(
            credentials=context.credentials,
            obj=status,
        ).unwrap()
        return res

    @service_method(
        path="code_status.update",
        name="update",
        roles=ADMIN_ROLE_LEVEL,
        autosplat=["code_update"],
        unwrap_on_success=False,
    )
    def update(
        self, context: AuthedServiceContext, code_update: CodeStatusUpdate
    ) -> SyftSuccess:
        existing_status = self.stash.get_by_uid(
            context.credentials, uid=code_update.id
        ).unwrap()
        server_identity = ServerIdentity.from_server(context.server)
        existing_status.status_dict[server_identity] = code_update.decision

        res = self.stash.update(context.credentials, existing_status).unwrap()
        return SyftSuccess(message="UserCode updated successfully", value=res)

    @service_method(
        path="code_status.get_by_uid", name="get_by_uid", roles=GUEST_ROLE_LEVEL
    )
    def get_status(
        self, context: AuthedServiceContext, uid: UID
    ) -> UserCodeStatusCollection:
        """Get the status of a user code item"""
        return self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

    @service_method(path="code_status.get_all", name="get_all", roles=ADMIN_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> list[UserCodeStatusCollection]:
        """Get all user code item statuses"""
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(
        path="code_status.remove",
        name="remove",
        roles=ADMIN_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def remove(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        """Remove a user code item status"""
        self.stash.delete_by_uid(context.credentials, uid=uid).unwrap()
        return SyftSuccess(message=f"{uid} successfully deleted", value=uid)


TYPE_TO_SERVICE[UserCodeStatusCollection] = UserCodeStatusService
