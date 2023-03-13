# future
from __future__ import annotations

# third party
from typing_extensions import Self

# relative
from ..new.credentials import SyftSigningKey
from ..new.document_store import StoreConfig
from ..new.node import NewNode
from .serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .uid import UID


@serializable(recursive_serde=True)
class WorkerSettings(SyftObject):
    __canonical_name__ = "WorkerSettings"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    signing_key: SyftSigningKey
    document_store_config: StoreConfig
    action_store_config: StoreConfig

    @staticmethod
    def from_node(node: NewNode) -> Self:
        return WorkerSettings(
            id=node.id,
            name=node.name,
            signing_key=node.signing_key,
            document_store_config=node.document_store_config,
            action_store_config=node.action_store_config,
        )

    def __hash__(self) -> int:
        return (
            hash(self.id)
            + hash(self.name)
            + hash(self.signing_key)
            + hash(self.document_store_config)
            + hash(self.action_store_config)
        )
