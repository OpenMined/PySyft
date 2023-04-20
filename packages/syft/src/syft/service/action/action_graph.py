# stdlib
from enum import Enum
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type

# third party
import networkx as nx
import pydantic
from typing_extensions import Self

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...store.document_store import StoreClientConfig
from ...store.document_store import StoreConfig
from ...store.locks import LockingConfig
from ...store.locks import NoLockingConfig
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from .action_object import Action


@serializable()
class ActionStatus(Enum):
    PROCESSING = 0
    DONE = 1
    FAILED = 2


@serializable()
class ActionGraphNode(SyftObject):
    __canonical_name__ = "ActionGraphNode"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    action: Optional[Action]
    status: ActionStatus = ActionStatus.PROCESSING
    retry: int = 0
    created_at: Optional[DateTime]
    credentials: SyftVerifyKey

    @pydantic.validator("created_at", pre=True, always=True)
    def make_result_id(cls, v: Optional[DateTime]) -> DateTime:
        return DateTime.now() if v is None else v

    @staticmethod
    def from_action(action: Action, credentials: SyftVerifyKey):
        return ActionGraphNode(id=action.id, action=action, credentials=credentials)

    def __hash__(self):
        return self.action.syft_history_hash

    def __eq__(self, other: Self):
        if not isinstance(other, ActionGraphNode):
            raise NotImplementedError(
                "Comparisions can be made with ActionGraphNode type objects only."
            )
        return hash(self) == hash(other)

    def __repr__(self):
        return self._repr_debug_()


@serializable()
class BaseGraphStore:
    graph_type: Any
    client_config: Optional[StoreClientConfig]

    def set(self, node: Any) -> None:
        raise NotImplementedError

    def delete(self, node: Any) -> None:
        raise NotImplementedError

    def find_neighbors(self, node: Any) -> List[Any]:
        raise NotImplementedError

    def update(self, node: Any) -> None:
        raise NotImplementedError

    def add_edge(self, parent: Any, child: Any) -> None:
        raise NotImplementedError

    def remove_edge(self, parent: Any, child: Any) -> None:
        raise NotImplementedError

    def nodes(self) -> Any:
        raise NotImplementedError

    def edges(self) -> Any:
        raise NotImplementedError

    def visualize(self) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def is_parent(self, parent: Any, child: Any):
        raise NotImplementedError


class InMemoryStoreClientConfig(StoreClientConfig):
    def __init__(self, path: Optional[str]):
        if path is None:
            self.file_path = Path(tempfile.gettempdir()) / "action_graph.bytes"
        else:
            self.file_path = path


@serializable()
class NetworkXBackingStore(BaseGraphStore):
    def __init__(self, store_config: StoreConfig) -> None:
        self.file_path = store_config.client_config.file_path

        if os.path.exists(self.file_path):
            self._db = self.load_from_path(self.file_path)
        else:
            self._db = nx.DiGraph()

    @property
    def db(self) -> nx.Graph:
        return self._db

    def set(self, node: ActionGraphNode) -> None:
        self.db.add_node(node)

    def delete(self, node: ActionGraphNode) -> None:
        self.graph.remove_node(node)

    def find_neighbors(self, node: ActionGraphNode) -> List[ActionGraphNode]:
        return self.graph.neighbors(node)

    def update(self, updated_node: ActionGraphNode) -> None:
        self.graph.update(updated_node)

    def add_edge(self, parent: ActionGraphNode, child: ActionGraphNode) -> None:
        self.graph.add_edge(parent, child)

    def remove_edge(self, parent: ActionGraphNode, child: ActionGraphNode) -> None:
        self.graph.remove_edge(parent, child)

    def visualize(self) -> None:
        return nx.draw_networkx(self.graph, with_labels=True)

    def nodes(self) -> Iterable:
        return self.graph.nodes()

    def edges(self) -> Iterable:
        return self.graph.edges()

    def save(self) -> None:
        bytes = _serialize(self.graph, to_bytes=True)
        with open(str(self.path), "wb") as f:
            f.write(bytes)

    def load_from_path(file_path: str) -> None:
        with open(file_path, "rb") as f:
            bytes = f.read()
        return _deserialize(blob=bytes, from_bytes=True)

    def is_parent(self, parent: ActionGraphNode, child: ActionGraphNode) -> bool:
        parents = list(self.graph.predecessors(child))
        return parent in parents


class InMemoryGraphConfig(StoreConfig):
    store_type: Type[BaseGraphStore] = NetworkXBackingStore
    client_config: StoreClientConfig = InMemoryStoreClientConfig
    locking_config: LockingConfig = NoLockingConfig()


class ActionGraphStore:
    pass


class InMemoryActionGraphStore(ActionGraphStore):
    def __init__(self, store_config: StoreConfig):
        self.store_config = store_config
        self.graph = self.store_config.store_type(self.store_config)

    def set(self, action: Action, credentials: SyftVerifyKey):
        #     node = ActionGraphNode.from_action(action, credentials)
        #     self._search_parents_for(node)
        #     self.graph.set(node)
        pass

    def get(self, action: Action, credentials: SyftVerifyKey):
        pass

    def delete(self, action: Action, credentials: SyftVerifyKey):
        pass

    def update(self, action: Action, credentials: SyftVerifyKey):
        pass
