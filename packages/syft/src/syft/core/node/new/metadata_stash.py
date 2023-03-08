# stdlib
from typing import List
from typing import Optional

# third party
from result import Result

# relative
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .document_store import BaseUIDStoreStash
from .document_store import DocumentStore
from .document_store import PartitionKey
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .node_metadata import NodeMetadata

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=List[UID])


@instrument
@serializable(recursive_serde=True)
class MetadataStash(BaseUIDStoreStash):
    object_type = NodeMetadata
    settings: PartitionSettings = PartitionSettings(
        name=NodeMetadata.__canonical_name__, object_type=NodeMetadata
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(self, metadata: NodeMetadata) -> Result[NodeMetadata, str]:
        return self.check_type(metadata, self.object_type).and_then(super().set)

    def update(self, metadata: NodeMetadata) -> Result[NodeMetadata, str]:
        return self.check_type(metadata, self.object_type).and_then(super().update)

    def get(self) -> Result[Optional[NodeMetadata], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(qks=qks)
