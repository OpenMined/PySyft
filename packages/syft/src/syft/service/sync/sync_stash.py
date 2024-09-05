# stdlib
import threading

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store_errors import StashException
from ...types.datetime import DateTime
from ...types.result import as_result
from ..context import AuthedServiceContext
from .sync_state import SyncState

OrderByDatePartitionKey = PartitionKey(key="created_at", type_=DateTime)


@serializable(canonical_name="SyncStash", version=1)
class SyncStash(NewBaseUIDStoreStash):
    object_type = SyncState
    settings: PartitionSettings = PartitionSettings(
        name=SyncState.__canonical_name__,
        object_type=SyncState,
    )

    def __init__(self, store: DocumentStore):
        super().__init__(store)
        self.store = store
        self.settings = self.settings
        self._object_type = self.object_type
        self.last_state: SyncState | None = None

    @as_result(StashException)
    def get_latest(self, context: AuthedServiceContext) -> SyncState | None:
        if self.last_state is not None:
            return self.last_state
        all_states = self.get_all(
            credentials=context.server.verify_key,  # type: ignore
            order_by=OrderByDatePartitionKey,
        ).unwrap()

        if len(all_states) > 0:
            self.last_state = all_states[-1]
            return all_states[-1]
        return None

    def unwrap_set(self, context: AuthedServiceContext, item: SyncState) -> SyncState:
        return super().set(context, item).unwrap()

    @as_result(StashException)
    def set(  # type: ignore
        self,
        context: AuthedServiceContext,
        item: SyncState,
        **kwargs,
    ) -> SyncState:
        self.last_state = item

        # use threading
        threading.Thread(
            target=self.unwrap_set,
            args=(
                context,
                item,
            ),
            kwargs=kwargs,
        ).start()
        return item
