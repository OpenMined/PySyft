# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from .vpn import VPNPeer


@serializable()
class VPNStash(BaseUIDStoreStash):
    object_type = VPNPeer
    settings: PartitionSettings = PartitionSettings(
        name=VPNPeer.__canonical_name__, object_type=VPNPeer
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
