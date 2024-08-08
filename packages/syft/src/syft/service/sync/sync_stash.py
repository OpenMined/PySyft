# stdlib

# stdlib

# stdlib
import threading

# third party
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.datetime import DateTime
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from .sync_state import SyncState

OrderByDatePartitionKey = PartitionKey(key="created_at", type_=DateTime)


@instrument
@serializable(canonical_name="SyncStash", version=1)
class SyncStash(BaseUIDStoreStash):
    object_type = SyncState
    settings: PartitionSettings = PartitionSettings(
        name=SyncState.__canonical_name__,
        object_type=SyncState,
    )

    def __init__(self, store: DocumentStore):
        super().__init__(store=store)
        self.store = store
        self.settings = self.settings
        self._object_type = self.object_type
        self.last_state: SyncState | None = None

    def get_latest(
        self, context: AuthedServiceContext
    ) -> Result[SyncState | None, str]:
        # print("SyncStash.get_latest called")
        if self.last_state is not None:
            return Ok(self.last_state)
        all_states = self.get_all(
            credentials=context.server.verify_key,  # type: ignore
            order_by=OrderByDatePartitionKey,
        )

        if all_states.is_err():
            return all_states

        all_states = all_states.ok()
        if len(all_states) > 0:
            self.last_state = all_states[-1]
            return Ok(all_states[-1])
        return Ok(None)

    def set(  # type: ignore
        self,
        context: AuthedServiceContext,
        item: SyncState,
        **kwargs,
    ) -> Result[SyncState, str]:
        self.last_state = item

        # use threading
        threading.Thread(
            target=super().set,
            args=(
                context,
                item,
            ),
            kwargs=kwargs,
        ).start()
        return Ok(item)
