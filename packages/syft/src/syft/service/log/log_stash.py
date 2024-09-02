# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionSettings
from .log import SyftLog


@serializable(canonical_name="LogStash", version=1)
class LogStash(NewBaseUIDStoreStash):
    object_type = SyftLog
    settings: PartitionSettings = PartitionSettings(
        name=SyftLog.__canonical_name__, object_type=SyftLog
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
