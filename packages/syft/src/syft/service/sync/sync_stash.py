# stdlib

# stdlib

# stdlib

# third party
from result import Ok
from result import Result
from syft.store.db.sqlite_db import DBManager

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.datetime import DateTime
from ...util.telemetry import instrument
from .sync_state import SyncState

OrderByDatePartitionKey = PartitionKey(key="created_at", type_=DateTime)


@instrument
@serializable(canonical_name="SyncStash", version=1)
class SyncStash(ObjectStash[SyncState]):
    settings: PartitionSettings = PartitionSettings(
        name=SyncState.__canonical_name__,
        object_type=SyncState,
    )

    def __init__(self, store: DBManager) -> None:
        super().__init__(store)
        self.last_state: SyncState | None = None

    def get_latest(self, credentials: SyftVerifyKey) -> Result[SyncState | None, str]:
        if self.last_state is not None:
            return Ok(self.last_state)

        states_or_err = self.get_all(
            credentials=credentials,
            sort_order="desc",
            limit=1,
        )

        if states_or_err.is_err():
            return states_or_err

        last_state = states_or_err.ok()
        if len(last_state) > 0:
            self.last_state = last_state[0]
            return Ok(last_state[0])
        return Ok(None)
