# future
from __future__ import annotations

# third party
from typing_extensions import Self

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ..new.credentials import SyftSigningKey
from ..new.document_store import StoreConfig
from ..new.node import NewNode


@serializable(recursive_serde=True)
class WorkerSettings(SyftObject):
    __canonical_name__ = "WorkerSettings"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    signing_key: SyftSigningKey
    store_config: StoreConfig

    @staticmethod
    def from_node(node: NewNode) -> Self:
        return WorkerSettings(
            id=node.id,
            name=node.name,
            signing_key=node.signing_key,
            store_config=node.store_config,
        )

    def __hash__(self) -> int:
        return (
            hash(self.id)
            + hash(self.name)
            + hash(self.signing_key)
            + hash(self.store_config)
        )
