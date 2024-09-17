# stdlib
from collections.abc import Callable
import inspect
import time

# relative
from ...serde.serializable import serializable
from ...server.worker_settings import WorkerSettings
from ...store.db.db import DBManager
from ...types.errors import SyftException
from ...types.uid import UID
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..code.user_code import UserCode
from ..context import AuthedServiceContext
from ..queue.queue_stash import ActionQueueItem
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .job_stash import Job
from .job_stash import JobStash
from .job_stash import JobStatus


def wait_until(predicate: Callable[[], bool], timeout: int = 10) -> SyftSuccess:
    start = time.time()
    code_string = inspect.getsource(predicate).strip()
    while time.time() - start < timeout:
        if predicate():
            return SyftSuccess(message=f"Predicate {code_string} is True")
        time.sleep(1)
    raise SyftException(public_message=f"Timeout reached for predicate {code_string}")


@serializable(canonical_name="JobService", version=1)
class JobService(AbstractService):
    stash: JobStash

    def __init__(self, store: DBManager) -> None:
        self.stash = JobStash(store=store)

    @service_method(
        path="job.get",
        name="get",
        roles=GUEST_ROLE_LEVEL,
    )
    def get(self, context: AuthedServiceContext, uid: UID) -> Job:
        return self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

    @service_method(path="job.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> list[Job]:
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(
        path="job.get_by_user_code_id",
        name="get_by_user_code_id",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_user_code_id(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> list[Job]:
        return self.stash.get_by_user_code_id(
            context.credentials, user_code_id
        ).unwrap()

    @service_method(
        path="job.delete",
        name="delete",
        roles=ADMIN_ROLE_LEVEL,
    )
    def delete(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        self.stash.delete_by_uid(context.credentials, uid).unwrap()
        return SyftSuccess(message="Great Success!")

    @service_method(
        path="job.get_by_result_id",
        name="get_by_result_id",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_by_result_id(self, context: AuthedServiceContext, result_id: UID) -> Job:
        return self.stash.get_by_result_id(context.credentials, result_id).unwrap()

    @service_method(
        path="job.restart",
        name="restart",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def restart(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        job = self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

        if job.parent_job_id is not None:
            raise SyftException(
                public_message="Not possible to restart subjobs. Please restart the parent job."
            )
        if job.status == JobStatus.PROCESSING:
            raise SyftException(
                public_message="Jobs in progress cannot be restarted. "
                "Please wait for completion or cancel the job via .cancel() to proceed."
            )

        job.status = JobStatus.CREATED
        self.update(context=context, job=job).unwrap()

        task_uid = UID()
        worker_settings = WorkerSettings.from_server(context.server)

        # TODO, fix return type of get_worker_pool_ref_by_name
        worker_pool_ref = context.server.get_worker_pool_ref_by_name(
            context.credentials
        )
        queue_item = ActionQueueItem(
            id=task_uid,
            server_uid=context.server.id,
            syft_client_verify_key=context.credentials,
            syft_server_location=context.server.id,
            job_id=job.id,
            worker_settings=worker_settings,
            args=[],
            kwargs={"action": job.action},
            worker_pool=worker_pool_ref,
        )

        context.server.queue_stash.set_placeholder(
            context.credentials, queue_item
        ).unwrap()

        self.stash.set(context.credentials, job).unwrap()
        context.server.services.log.restart(context, job.log_id)

        return SyftSuccess(message="Great Success!")

    @service_method(
        path="job.update",
        name="update",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def update(self, context: AuthedServiceContext, job: Job) -> SyftSuccess:
        res = self.stash.update(context.credentials, obj=job).unwrap()
        return SyftSuccess(message="Job updated!", value=res)

    def _kill(self, context: AuthedServiceContext, job: Job) -> SyftSuccess:
        # set job and subjobs status to TERMINATING
        # so that MonitorThread can kill them
        job.status = JobStatus.TERMINATING
        res = self.stash.update(context.credentials, obj=job).unwrap()
        results = [res]

        # attempt to kill all subjobs
        subjobs = self.stash.get_by_parent_id(context.credentials, uid=job.id).unwrap()
        if subjobs is not None:
            for subjob in subjobs:
                subjob.status = JobStatus.TERMINATING
                res = self.stash.update(context.credentials, obj=subjob).unwrap()
                results.append(res)

        # wait for job and subjobs to be killed by MonitorThread
        wait_until(lambda: job.fetched_status == JobStatus.INTERRUPTED)
        wait_until(
            lambda: all(
                subjob.fetched_status == JobStatus.INTERRUPTED for subjob in job.subjobs
            )
        )

        return SyftSuccess(message="Job killed successfully!")

    @service_method(
        path="job.kill",
        name="kill",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def kill(self, context: AuthedServiceContext, id: UID) -> SyftSuccess:
        job = self.stash.get_by_uid(context.credentials, uid=id).unwrap()
        if job.parent_job_id is not None:
            raise SyftException(
                public_message="Not possible to cancel subjobs. To stop execution, please cancel the parent job."
            )
        if job.status != JobStatus.PROCESSING:
            raise SyftException(public_message="Job is not running")
        if job.job_pid is None:
            raise SyftException(
                public_message="Job termination disabled in dev mode. "
                "Set 'dev_mode=False' or 'thread_workers=False' to enable."
            )

        return self._kill(context, job)

    @service_method(
        path="job.get_subjobs",
        name="get_subjobs",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_subjobs(self, context: AuthedServiceContext, uid: UID) -> list[Job]:
        return self.stash.get_by_parent_id(context.credentials, uid=uid).unwrap()

    @service_method(
        path="job.get_active", name="get_active", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_active(self, context: AuthedServiceContext) -> list[Job]:
        return self.stash.get_active(context.credentials).unwrap()

    @service_method(
        path="job.add_read_permission_job_for_code_owner",
        name="add_read_permission_job_for_code_owner",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def add_read_permission_job_for_code_owner(
        self, context: AuthedServiceContext, job: Job, user_code: UserCode
    ) -> None:
        permission = ActionObjectPermission(
            job.id, ActionPermission.READ, user_code.user_verify_key
        )
        # TODO: make add_permission wrappable
        return self.stash.add_permission(permission=permission).unwrap()

    @service_method(
        path="job.add_read_permission_log_for_code_owner",
        name="add_read_permission_log_for_code_owner",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def add_read_permission_log_for_code_owner(
        self, context: AuthedServiceContext, log_id: UID, user_code: UserCode
    ) -> None:
        return context.server.services.log.stash.add_permission(
            ActionObjectPermission(
                log_id, ActionPermission.READ, user_code.user_verify_key
            )
        ).unwrap()

    @service_method(
        path="job.create_job_for_user_code_id",
        name="create_job_for_user_code_id",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def create_job_for_user_code_id(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        result: ActionObject | None = None,
        log_stdout: str = "",
        log_stderr: str = "",
        status: JobStatus = JobStatus.CREATED,
        add_code_owner_read_permissions: bool = True,
    ) -> Job:
        is_resolved = status in [JobStatus.COMPLETED, JobStatus.ERRORED]
        job = Job(
            id=UID(),
            server_uid=context.server.id,
            action=None,
            result=result,
            status=status,
            parent_id=None,
            log_id=UID(),
            job_pid=None,
            user_code_id=user_code_id,
            resolved=is_resolved,
        )
        user_code = context.server.services.user_code.get_by_uid(
            context=context, uid=user_code_id
        )

        # The owner of the code should be able to read the job
        self.stash.set(context.credentials, job).unwrap()
        context.server.services.log.add(
            context,
            job.log_id,
            job.id,
            stdout=log_stdout,
            stderr=log_stderr,
        )

        if add_code_owner_read_permissions:
            self.add_read_permission_job_for_code_owner(context, job, user_code)
            self.add_read_permission_log_for_code_owner(context, job.log_id, user_code)

        return job


TYPE_TO_SERVICE[Job] = JobService
