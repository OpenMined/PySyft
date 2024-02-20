from typing import Optional, Union

from syft.service.context import AuthedServiceContext
from syft.service.response import SyftError

from ...types.datetime import DateTime
from ...store.document_store import PartitionKey, PartitionSettings, BaseUIDStoreStash
from .sync_state import SyncState
from ...util.telemetry import instrument
from ...serde.serializable import serializable

OrderByDatePartitionKey = PartitionKey(key="created_at", type_=DateTime)


@instrument
@serializable()
class SyncStash(BaseUIDStoreStash):
    object_type = SyncState
    settings: PartitionSettings = PartitionSettings(
        name=SyncState.__canonical_name__, object_type=SyncState
    )

    def __init__(self, store):
        super().__init__(store)
        self.store = store
        self.settings = self.settings
        self._object_type = self.object_type

    def get_latest(
        self, context: AuthedServiceContext
    ) -> Union[Optional[SyncState], SyftError]:
        all_states = self.get_all(
            credentials=context.node.verify_key,
            order_by=OrderByDatePartitionKey,
            limit=1,
        )

        if isinstance(all_states, str):
            return SyftError(message=all_states)

        if len(all_states) > 0:
            return all_states[0]
        return None
