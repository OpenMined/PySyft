# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...types.uid import UID
from ..action.action_permissions import StoragePermission
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .log import SyftLog
from .log_stash import LogStash


@serializable(canonical_name="LogService", version=1)
class LogService(AbstractService):
    stash: LogStash

    def __init__(self, store: DBManager) -> None:
        self.stash = LogStash(store=store)

    @service_method(path="log.add", name="add", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def add(
        self,
        context: AuthedServiceContext,
        uid: UID,
        job_id: UID,
        stdout: str = "",
        stderr: str = "",
    ) -> SyftSuccess:
        new_log = SyftLog(id=uid, job_id=job_id, stdout=stdout, stderr=stderr)
        return self.stash.set(context.credentials, new_log).unwrap()

    @service_method(path="log.append", name="append", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def append(
        self,
        context: AuthedServiceContext,
        uid: UID,
        new_str: str = "",
        new_err: str = "",
    ) -> SyftSuccess:
        new_log = self.stash.get_by_uid(context.credentials, uid).unwrap()
        if new_str:
            new_log.append(new_str)

        if new_err:
            new_log.append_error(new_err)

        self.stash.update(context.credentials, new_log).unwrap()
        return SyftSuccess(message="Log Append successful!")

    @service_method(path="log.get", name="get", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get(self, context: AuthedServiceContext, uid: UID) -> SyftLog:
        return self.stash.get_by_uid(context.credentials, uid).unwrap()

    @service_method(
        path="log.get_stdout", name="get_stdout", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_stdout(self, context: AuthedServiceContext, uid: UID) -> str:
        result = self.get(context, uid)
        return result.stdout

    @service_method(path="log.get_stderr", name="get_stderr", roles=ADMIN_ROLE_LEVEL)
    def get_stderr(self, context: AuthedServiceContext, uid: UID) -> str:
        result = self.get(context, uid)
        return result.stderr

    @service_method(path="log.restart", name="restart", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def restart(
        self,
        context: AuthedServiceContext,
        uid: UID,
    ) -> SyftSuccess:
        log = self.stash.get_by_uid(context.credentials, uid).unwrap()
        log.restart()
        self.stash.update(context.credentials, log).unwrap()
        return SyftSuccess(message="Log Restart successful!")

    @service_method(path="log.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> list[SyftLog]:
        return self.stash.get_all(context.credentials).unwrap()  # type: ignore

    @service_method(path="log.delete", name="delete", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def delete(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        self.stash.delete_by_uid(context.credentials, uid).unwrap()
        return SyftSuccess(message=f"log {uid} succesfully deleted")

    @service_method(
        path="log.has_storage_permission",
        name="has_storage_permission",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def has_storage_permission(self, context: AuthedServiceContext, uid: UID) -> bool:
        permission = StoragePermission(uid=uid, server_uid=context.server.id)
        result = self.stash.has_storage_permission(permission)
        return result


TYPE_TO_SERVICE[SyftLog] = LogService
