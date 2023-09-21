# stdlib
from typing import List
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .job_stash import Job
from .job_stash import JobStash


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
            return res.ok()

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
