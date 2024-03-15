# stdlib

# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.datetime import DateTime
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from .sync_state import SyncState

OrderByDatePartitionKey = PartitionKey(key="created_at", type_=DateTime)


@instrument
@serializable()
class SyncStash(BaseUIDStoreStash):
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

    def get_latest(self, context: AuthedServiceContext) -> SyncState | None | SyftError:
        all_states = self.get_all(
            credentials=context.node.verify_key,  # type: ignore
            order_by=OrderByDatePartitionKey,
        )

        if all_states.is_err():
            return SyftError(message=all_states.err())

        all_states = all_states.ok()
        if len(all_states) > 0:
            return all_states[-1]
        return None
