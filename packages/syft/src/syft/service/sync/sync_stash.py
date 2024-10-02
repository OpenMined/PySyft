# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.db import DBManager
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import StashException
from ...types.result import as_result
from .sync_state import SyncState


@serializable(canonical_name="SyncStash", version=1)
class SyncStash(ObjectStash[SyncState]):
    def __init__(self, store: DBManager) -> None:
        super().__init__(store)
        self.last_state: SyncState | None = None

    @as_result(StashException)
    def get_latest(self, credentials: SyftVerifyKey) -> SyncState | None:
        if self.last_state is not None:
            return self.last_state

        states = self.get_all(
            credentials=credentials,
            order_by="created_at",
            sort_order="desc",
            limit=1,
        ).unwrap()

        if len(states) > 0:
            return states[0]
        return None
