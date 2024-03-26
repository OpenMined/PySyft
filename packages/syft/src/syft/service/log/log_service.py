# stdlib

# third party
from result import Ok

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .log import SyftLog
from .log_stash import LogStash


@instrument
@serializable()
class LogService(AbstractService):
    store: DocumentStore
    stash: LogStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = LogStash(store=store)

    @service_method(path="log.add", name="add", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def add(
        self, context: AuthedServiceContext, uid: UID, job_id: UID
    ) -> SyftSuccess | SyftError:
        new_log = SyftLog(id=uid, job_id=job_id)
        result = self.stash.set(context.credentials, new_log)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result

    @service_method(path="log.append", name="append", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def append(
        self,
        context: AuthedServiceContext,
        uid: UID,
        new_str: str = "",
        new_err: str = "",
    ) -> SyftSuccess | SyftError:
        result = self.stash.get_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=str(result.err()))
        new_log = result.ok()
        if new_str:
            new_log.append(new_str)

        if new_err:
            new_log.append_error(new_err)

        result = self.stash.update(context.credentials, new_log)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Log Append successful!")

    @service_method(path="log.get", name="get", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess | SyftError:
        result = self.stash.get_by_uid(context.credentials, uid)

        if result.is_err():
            return SyftError(message=str(result.err()))

        return result

    @service_method(
        path="log.get_stdout", name="get_stdout", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_stdout(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        result = self.stash.get_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=str(result.err()))

        return Ok(result.ok().stdout)

    @service_method(path="log.restart", name="restart", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def restart(
        self,
        context: AuthedServiceContext,
        uid: UID,
    ) -> SyftSuccess | SyftError:
        result = self.stash.get_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=str(result.err()))

        log = result.ok()
        log.restart()
        result = self.stash.update(context.credentials, log)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Log Restart successful!")

    @service_method(path="log.get_error", name="get_error", roles=ADMIN_ROLE_LEVEL)
    def get_error(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        result = self.stash.get_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=str(result.err()))

        return Ok(result.ok().stderr)

    @service_method(path="log.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> SyftSuccess | SyftError:
        result = self.stash.get_all(context.credentials)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path="log.delete", name="delete", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def delete(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        result = self.stash.delete_by_uid(context.credentials, uid)
        if result.is_ok():
            return result.ok()
        else:
            return SyftError(message=result.err())

    @service_method(
        path="log.has_storage_permission",
        name="has_storage_permission",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def has_storage_permission(self, context: AuthedServiceContext, uid: UID) -> bool:
        result = self.stash.has_storage_permission(
            uid,
        )

        return result


TYPE_TO_SERVICE[SyftLog] = LogService
