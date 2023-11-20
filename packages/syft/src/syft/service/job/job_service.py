# stdlib
from typing import List
from typing import Union

# third party
import psutil

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
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
        if job.job_pid is not None:
            process = psutil.Process(job.job_pid)
            process.terminate()
            job.status = JobStatus.INTERRUPTED
            job.resolved = True
            res = self.stash.update(context.credentials, obj=job)
            if res.is_err():
                return SyftError(message=res.err())

        res = res.ok()
        return SyftSuccess(message="Great Success!")

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
