# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from .log import SyftLogV2


@instrument
@serializable()
class LogStash(BaseUIDStoreStash):
    object_type = SyftLogV2
    settings: PartitionSettings = PartitionSettings(
        name=SyftLogV2.__canonical_name__, object_type=SyftLogV2
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
