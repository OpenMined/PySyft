# relative
from ...serde.serializable import serializable
from ...store.db.base_stash import ObjectStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from .log import SyftLog


@instrument
@serializable(canonical_name="LogStash", version=1)
class LogStash(ObjectStash[SyftLog]):
    object_type = SyftLog
    settings: PartitionSettings = PartitionSettings(
        name=SyftLog.__canonical_name__, object_type=SyftLog
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
