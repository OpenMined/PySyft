# stdlib

# stdlib

# third party
from result import as_result

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store_errors import StashException
from ...types.datetime import DateTime
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from .sync_state import SyncState

OrderByDatePartitionKey = PartitionKey(key="created_at", type_=DateTime)


@instrument
@serializable()
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

    @as_result(StashException)
    def get_latest(self, context: AuthedServiceContext) -> SyncState | None:
        all_states = self.get_all(
            credentials=context.node.verify_key,  # type: ignore
            order_by=OrderByDatePartitionKey,
        ).unwrap()

        if len(all_states) > 0:
            return all_states[-1]
        return None
