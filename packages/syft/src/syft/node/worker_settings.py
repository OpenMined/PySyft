# future
from __future__ import annotations

# third party
from typing_extensions import Self

# relative
from ..abstract_node import AbstractNode
from ..abstract_node import NodeSideType
from ..abstract_node import NodeType
from ..node.credentials import SyftSigningKey
from ..serde.serializable import serializable
from ..service.queue.base_queue import QueueConfig
from ..store.blob_storage import BlobStorageConfig
from ..store.document_store import StoreConfig
from ..types.syft_object import SYFT_OBJECT_VERSION_3
from ..types.syft_object import SyftObject
from ..types.uid import UID


@serializable()
class WorkerSettings(SyftObject):
    __canonical_name__ = "WorkerSettings"
    __version__ = SYFT_OBJECT_VERSION_3

    id: UID
    name: str
    node_type: NodeType
    node_side_type: NodeSideType
    signing_key: SyftSigningKey
    document_store_config: StoreConfig
    action_store_config: StoreConfig
    blob_store_config: BlobStorageConfig | None = None
    queue_config: QueueConfig | None = None

    @classmethod
    def from_node(cls, node: AbstractNode) -> Self:
        if node.node_side_type:
            node_side_type: str = node.node_side_type.value
        else:
            node_side_type = NodeSideType.HIGH_SIDE
        return cls(
            id=node.id,
            name=node.name,
            node_type=node.node_type,
            signing_key=node.signing_key,
            document_store_config=node.document_store_config,
            action_store_config=node.action_store_config,
            node_side_type=node_side_type,
            blob_store_config=node.blob_store_config,
            queue_config=node.queue_config,
        )
