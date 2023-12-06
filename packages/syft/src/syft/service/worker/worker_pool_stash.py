# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from .worker_pool import WorkerPool


@serializable()
class SyftWorkerPoolStash(BaseUIDStoreStash):
    object_type = WorkerPool
    settings: PartitionSettings = PartitionSettings(
        name=WorkerPool.__canonical_name__,
        object_type=WorkerPool,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
