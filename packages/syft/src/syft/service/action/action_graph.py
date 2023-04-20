# stdlib
from enum import Enum
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
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...store.document_store import StoreClientConfig
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

    @pydantic.validator("created_at", pre=True, always=True)
    def make_result_id(cls, v: Optional[DateTime]) -> DateTime:
        return DateTime.now() if v is None else v

    @staticmethod
    def from_action(action: Action):
        return ActionGraphNode(id=action.id, action=action)

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


# class NetworkXClientConfig(StoreClientConfig):
#     __canonical_name__ = "NetworkXClientConfig"
#     __version__ = SYFT_OBJECT_VERSION_1

# class Neo4JClientConfig(StoreClientConfig):
#     __canonical_name__ = "Neo4JClientConfig"
#     __version__ = SYFT_OBJECT_VERSION_1

#     db_name: str
#     host: str
#     port: int


@serializable()
class BaseGraphClient:
    graph_type: Any
    client_config: Optional[StoreClientConfig]

    @staticmethod
    def init_graph() -> Self:
        raise NotImplementedError

    def add_node(self, node: ActionGraphNode) -> None:
        raise NotImplementedError

    def remove_node(self, node: ActionGraphNode) -> None:
        raise NotImplementedError

    def find_neighbors(self, node: ActionGraphNode) -> List[ActionGraphNode]:
        raise NotImplementedError

    def update(self, updated_node: ActionGraphNode) -> None:
        raise NotImplementedError

    def add_edge(self, parent: ActionGraphNode, child: ActionGraphNode) -> None:
        raise NotImplementedError

    def remove_edge(self, parent: ActionGraphNode, child: ActionGraphNode) -> None:
        raise NotImplementedError

    def visualize(self) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def is_parent(self, parent: ActionGraphNode, child: ActionGraphNode):
        raise NotImplementedError


@serializable()
class InMemoryGraphClient(BaseGraphClient):
    graph_type: nx.DiGraph = nx.DiGraph
    graph: graph_type

    def __init__(self, path: Optional[str] = None):
        self.graph = nx.DiGraph()
        if path is None:
            self.path = Path(tempfile.gettempdir()) / "action_graph.bytes"
            # TODO: repace self.path with a name in the config class like in SQLiteStoreClientConfig
        else:
            self.path = path

    @staticmethod
    def init_graph(path: Optional[str] = None) -> Self:
        return InMemoryGraphClient(path)

    def add_node(self, node: ActionGraphNode) -> None:
        self.graph.add_node(node)

    def remove_node(self, node: ActionGraphNode) -> None:
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

    @property
    def nodes(self) -> Iterable:
        return self.graph.nodes()

    @property
    def edges(self) -> Iterable:
        return self.graph.edges()

    def save(self) -> None:
        print(self.path)
        bytes = _serialize(self.graph, to_bytes=True)
        with open(str(self.path), "wb") as f:
            f.write(bytes)

    def load(self) -> None:
        print(self.path)
        with open(str(self.path), "rb") as f:
            bytes = f.read()
        self.graph = _deserialize(blob=bytes, from_bytes=True)

    def is_parent(self, parent: ActionGraphNode, child: ActionGraphNode) -> bool:
        parents = list(self.graph.predecessors(child))
        return parent in parents


# class GraphClientConfig:
#    pass


@serializable()
class ActionGraph:
    def __init__(self, node_uid: UID, graph_client: Type[BaseGraphClient]):
        self.node_uid = node_uid
        self.client = graph_client.init_graph()

    def add_action(self, action: Action) -> None:
        # TODO: Handle Duplication
        node = ActionGraphNode.from_action(action)
        # self.client.add_node(node)
        self._search_parents_for(node)
        self.client.add_node(node)

    def _search_parents_for(self, node: ActionGraphNode) -> None:
        input_ids = []
        parents = set()
        if node.action.remote_self:
            input_ids.append(node.action.remote_self)
        input_ids.extend(node.action.args)
        input_ids.extend(node.action.kwargs.values())
        # search for parents in the existing nodes
        for _node in self.client.nodes:
            if _node.action.result_id in input_ids:
                parents.add(_node)

        for parent in parents:
            self.client.add_edge(parent, node)

    def remove_action(self, action: Action):
        node = ActionGraphNode.from_action(action)
        self.client.remove_node(node)

    def draw_graph(self):
        return self.client.visualize()

    @property
    def nodes(self):
        return self.client.nodes

    @property
    def edges(self):
        return self.client.edges

    def is_parent(self, parent: Action, child: Action) -> bool:
        parent_node: ActionGraphNode = ActionGraphNode.from_action(parent)
        child_node: ActionGraphNode = ActionGraphNode.from_action(child)
        return self.client.is_parent(parent_node, child_node)

    def save(self) -> None:
        self.client.save()

    def load(self) -> None:
        self.client.load()


# class ActionGraphVersion2:
#     node_uid: UID

#     def __init__(self, node_uid: UID):
#         self._graph = nx.DiGraph(name=node_uid)
#         self.node_uid = node_uid

#     def add(self, node: ActionGraphObject) -> None:
#         self._graph.add_node(node.id, data=node)

#     def add_relationship(
#         self, nodeA: ActionGraphObject, nodeB: ActionGraphObject
#     ) -> None:
#         self._graph.add_edge(nodeA, nodeB)

#     def remove_node(self, node: ActionGraphObject) -> None:
#         self._graph.remove_node(node)

#     def remove_edge(self, node: ActionGraphObject) -> None:
#         self._graph.remove_edge(node)

#     def neighbors_for(self, node: ActionGraphObject) -> List:
#         return list(self._graph.neighbors(node))

#     def visualize(self, arrows: bool = True) -> Any:
#         return nx.draw_networkx(self._graph)

#     def remove_all_nodes_from(self, node: ActionGraphObject):
#         all_adjacent_neighbors: list = []

#         def find_adjacent_neighbors(node: ActionGraphObject, neighbors: set):
#             if not self._graph.neighbors(node):
#                 return

#             my_neighbors = self._graph.neighbors(node)
#             for n in my_neighbors:
#                 if n not in neighbors:
#                     neighbors.add(n)
#                     self.find_adjacent_neighbors(n, neighbors)

#         find_adjacent_neighbors(node, all_adjacent_neighbors)

#         self._graph.remove_nodes_from(all_adjacent_neighbors)
