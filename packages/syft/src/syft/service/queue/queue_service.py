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
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .queue_stash import QueueItem
from .queue_stash import QueueStash
from .queue_stash import Status


@instrument
@serializable()
class QueueService(AbstractService):
    store: DocumentStore
    stash: QueueStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = QueueStash(store=store)

    @service_method(
        path="queue.get_subjobs",
        name="get_subjobs",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_subjobs(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[List[QueueItem], SyftError]:
        res = self.stash.get_by_parent_id(context.credentials, uid=uid)
        if res.is_err():
            return SyftError(message=res.err())
        else:
            return res.ok()

    @service_method(
        path="queue.get_by_job_id",
        name="get_by_job_id",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_job_id(
        self, context: AuthedServiceContext, job_id: UID
    ) -> Union[QueueItem, SyftError]:
        res = self.stash.get_by_job_id(context.credentials, job_id=job_id)
        if res.is_err():
            return SyftError(message=res.err())
        else:
            return res.ok()

    @service_method(
        path="queue.update_status",
        name="update_status",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def update_queue_status(
        self, context: AuthedServiceContext, uid: UID, status: Status
    ) -> Union[QueueItem, SyftError]:
        res = self.stash.get_by_uid(uid=uid)
        if res.is_err():
            return SyftError(
                message=f"Failed to get the QueueItem with id <{uid}>. Error: {res.err()}"
            )
        queue_item: QueueItem = res.ok()
        queue_item.status = status
        res = self.stash.update(credentials=context.credentials, obj=queue_item)
        if res.is_ok():
            return res.ok()
        return SyftError(
            message=f"Failed to update QueueItem with id <{queue_item.id}>. Error: {res.err()}"
        )
