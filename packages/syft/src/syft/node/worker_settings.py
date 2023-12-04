# future
from __future__ import annotations

# stdlib
from typing import Optional

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
from ..types.syft_migration import migrate
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SYFT_OBJECT_VERSION_2
from ..types.syft_object import SyftObject
from ..types.transforms import drop
from ..types.transforms import make_set_default
from ..types.uid import UID


@serializable()
class WorkerSettingsV1(SyftObject):
    __canonical_name__ = "WorkerSettings"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    node_type: NodeType
    node_side_type: NodeSideType
    signing_key: SyftSigningKey
    document_store_config: StoreConfig
    action_store_config: StoreConfig
    blob_store_config: Optional[BlobStorageConfig]


@serializable()
class WorkerSettings(SyftObject):
    __canonical_name__ = "WorkerSettings"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    name: str
    node_type: NodeType
    node_side_type: NodeSideType
    signing_key: SyftSigningKey
    document_store_config: StoreConfig
    action_store_config: StoreConfig
    blob_store_config: Optional[BlobStorageConfig]
    queue_config: Optional[QueueConfig]

    @staticmethod
    def from_node(node: AbstractNode) -> Self:
        return WorkerSettings(
            id=node.id,
            name=node.name,
            node_type=node.node_type,
            signing_key=node.signing_key,
            document_store_config=node.document_store_config,
            action_store_config=node.action_store_config,
            node_side_type=node.node_side_type.value,
            blob_store_config=node.blob_store_config,
            queue_config=node.queue_config,
        )


# queue_config


@migrate(WorkerSettings, WorkerSettingsV1)
def downgrade_workersettings_v2_to_v1():
    return [
        drop(["queue_config"]),
    ]


@migrate(WorkerSettingsV1, WorkerSettings)
def upgrade_workersettings_v1_to_v2():
    # relative
    from ..service.queue.zmq_queue import ZMQQueueConfig

    return [make_set_default("queue_config", ZMQQueueConfig())]
