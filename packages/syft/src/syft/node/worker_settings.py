# future
from __future__ import annotations

# third party
from typing_extensions import Self

# relative
from ..abstract_node import AbstractNode
from ..abstract_node import NodeType
from ..node.credentials import SyftSigningKey
from ..serde.serializable import serializable
from ..store.document_store import StoreConfig
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID


@serializable()
class WorkerSettings(SyftObject):
    __canonical_name__ = "WorkerSettings"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    node_type: NodeType
    signing_key: SyftSigningKey
    document_store_config: StoreConfig
    action_store_config: StoreConfig

    @staticmethod
    def from_node(node: AbstractNode) -> Self:
        return WorkerSettings(
            id=node.id,
            name=node.name,
            node_type=node.node_type,
            signing_key=node.signing_key,
            document_store_config=node.document_store_config,
            action_store_config=node.action_store_config,
        )
