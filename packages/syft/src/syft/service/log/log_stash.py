# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from .log import SyftLog


@instrument
@serializable()
class LogStash(BaseStash):
    object_type = SyftLog
    settings: PartitionSettings = PartitionSettings(
        name=SyftLog.__canonical_name__, object_type=SyftLog
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
