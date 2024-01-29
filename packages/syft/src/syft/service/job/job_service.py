# stdlib
from typing import List
from typing import Union

# relative
from ...node.worker_settings import WorkerSettings
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..queue.queue_stash import ActionQueueItem
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .job_stash import Job
from .job_stash import JobStash
from .job_stash import JobStatus


@instrument
@serializable()
class JobService(AbstractService):
    store: DocumentStore
    stash: JobStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = JobStash(store=store)

    @service_method(
        path="job.get",
        name="get",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[List[Job], SyftError]:
        res = self.stash.get_by_uid(context.credentials, uid=uid)
        if res.is_err():
            return SyftError(message=res.err())
        else:
            res = res.ok()
            return res

    @service_method(
        path="job.get_all",
        name="get_all",
    )
    def get_all(self, context: AuthedServiceContext) -> Union[List[Job], SyftError]:
        res = self.stash.get_all(context.credentials)
        if res.is_err():
            return SyftError(message=res.err())
        else:
            res = res.ok()
            return res

    @service_method(
        path="job.get_by_user_code_id",
        name="get_by_user_code_id",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_user_code_id(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> Union[List[Job], SyftError]:
        res = self.stash.get_by_user_code_id(context.credentials, user_code_id)
        if res.is_err():
            return SyftError(message=res.err())

        res = res.ok()
        return res

    @service_method(
        path="job.restart",
        name="restart",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def restart(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        res = self.stash.get_by_uid(context.credentials, uid=uid)
        if res.is_err():
            return SyftError(message=res.err())

        job = res.ok()
        job.status = JobStatus.CREATED
        self.update(context=context, job=job)

        task_uid = UID()
        worker_settings = WorkerSettings.from_node(context.node)

        queue_item = ActionQueueItem(
            id=task_uid,
            node_uid=context.node.id,
            syft_client_verify_key=context.credentials,
            syft_node_location=context.node.id,
            job_id=job.id,
            worker_settings=worker_settings,
            args=[],
            kwargs={"action": job.action},
        )

        context.node.queue_stash.set_placeholder(context.credentials, queue_item)
        context.node.job_stash.set(context.credentials, job)
        log_service = context.node.get_service("logservice")
        result = log_service.restart(context, job.log_id)

        if result.is_err():
            return SyftError(message=str(result.err()))

        return SyftSuccess(message="Great Success!")

    @service_method(
        path="job.update",
        name="update",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def update(
        self, context: AuthedServiceContext, job: Job
    ) -> Union[SyftSuccess, SyftError]:
        res = self.stash.update(context.credentials, obj=job)
        if res.is_err():
            return SyftError(message=res.err())
        res = res.ok()
        return SyftSuccess(message="Great Success!")

    @service_method(
        path="job.kill",
        name="kill",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def kill(
        self, context: AuthedServiceContext, id: UID
    ) -> Union[SyftSuccess, SyftError]:
        res = self.stash.get_by_uid(context.credentials, uid=id)
        if res.is_err():
            return SyftError(message=res.err())

        job = res.ok()
        if job.job_pid is not None and job.status == JobStatus.PROCESSING:
            job.status = JobStatus.INTERRUPTED
            res = self.stash.update(context.credentials, obj=job)
            if res.is_err():
                return SyftError(message=res.err())
            return SyftSuccess(message="Job killed successfully!")
        else:
            return SyftError(
                message="Job is not running or isn't running in multiprocessing mode."
                "Killing threads is currently not supported"
            )

    @service_method(
        path="job.get_subjobs",
        name="get_subjobs",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_subjobs(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[List[Job], SyftError]:
        res = self.stash.get_by_parent_id(context.credentials, uid=uid)
        if res.is_err():
            return SyftError(message=res.err())
        else:
            return res.ok()

    @service_method(
        path="job.get_active", name="get_active", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_active(self, context: AuthedServiceContext) -> Union[List[Job], SyftError]:
        res = self.stash.get_active(context.credentials)
        if res.is_err():
            return SyftError(message=res.err())
        return res.ok()

    @service_method(
        path="job.create_job_for_user_code_id",
        name="create_job_for_user_code_id",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def create_job_for_user_code_id(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> Union[Job, SyftError]:
        job = Job(
            id=UID(),
            node_uid=context.node.id,
            action=None,
            result_id=None,
            parent_id=None,
            log_id=UID(),
            job_pid=None,
            user_code_id=user_code_id,
        )

        user_code_service = context.node.get_service("usercodeservice")
        user_code = user_code_service.get_by_uid(context=context, uid=user_code_id)
        if isinstance(user_code, SyftError):
            return user_code

        # The owner of the code should be able to read the job
        permission = ActionObjectPermission(
            job.id, ActionPermission.READ, user_code.user_verify_key
        )
        self.stash.set(context.credentials, job, add_permissions=[permission])

        log_service = context.node.get_service("logservice")
        res = log_service.add(context, job.log_id)
        if isinstance(res, SyftError):
            return res
        # The owner of the code should be able to read the job log
        log_service.stash.add_permission(
            ActionObjectPermission(
                job.log_id, ActionPermission.READ, user_code.user_verify_key
            )
        )

        return job
