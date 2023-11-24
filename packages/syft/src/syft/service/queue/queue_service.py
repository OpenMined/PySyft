# stdlib
from typing import Any
from typing import List
from typing import Union

# syft absolute
from syft.service.queue.zmq_queue import ZMQConsumerView

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .queue_stash import QueueItem
from .queue_stash import QueueStash


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
        path="queue.consumers",
        name="get_all_consumers",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_all_consumers(
        self, context: AuthedServiceContext
    ) -> Union[list[ZMQConsumerView], SyftError]:
        try:
            node = context.node
            all_consumers: list[ZMQConsumerView] = []
            for _, consumers in node.queue_manager.consumers.items():
                for consumer in consumers:
                    all_consumers.append(ZMQConsumerView(consumer))
            return all_consumers
        except Exception as e:
            return SyftError(
                message=f"Something went wrong retrieving the workers: {e}"
            )
