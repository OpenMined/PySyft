# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...types.blob_storage import BlobStorageEntry


@serializable()
class BlobStorageStash(BaseUIDStoreStash):
    object_type = BlobStorageEntry
    settings: PartitionSettings = PartitionSettings(
        name=BlobStorageEntry.__canonical_name__, object_type=BlobStorageEntry
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
