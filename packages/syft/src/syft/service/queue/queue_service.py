# stdlib

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
        method_params = {
            "get": {'skip_method': True},
            "set": {'skip_method': True},
            "update": {'skip_method': True},
            "delete": {'skip_method': True}
        }
        super().__init__(method_params=method_params)


    @service_method(
        path="queue.get_subjobs",
        name="get_subjobs",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_subjobs(
        self, context: AuthedServiceContext, uid: UID
    ) -> list[QueueItem] | SyftError:
        res = self.stash.get_by_parent_id(context.credentials, uid=uid)
        if res.is_err():
            return SyftError(message=res.err())
        else:
            return res.ok()
