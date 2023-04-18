# stdlib
from enum import Enum
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
from ...serde.serializable import serializable
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

    def add_node(self, node: ActionGraphNode):
        raise NotImplementedError

    def remove_node(self, node: ActionGraphNode):
        raise NotImplementedError

    def find_neighbors(self, node: ActionGraphNode) -> List[ActionGraphNode]:
        raise NotImplementedError

    def update(self, updated_node: ActionGraphNode):
        raise NotImplementedError

    def add_edge(self, parent: ActionGraphNode, child: ActionGraphNode):
        raise NotImplementedError

    def remove_edge(self, parent: ActionGraphNode, child: ActionGraphNode):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

    def save(self):
        # TODO 🟡: Add functionality to save the graph
        pass


@serializable()
class InMemoryGraphClient(BaseGraphClient):
    graph_type: nx.DiGraph = nx.DiGraph

    def __init__(self):
        self.graph = nx.DiGraph()

    @staticmethod
    def init_graph() -> Self:
        return InMemoryGraphClient()

    def add_node(self, node: ActionGraphNode):
        self.graph.add_node(node)

    def remove_node(self, node: ActionGraphNode):
        self.graph.remove_node(node)

    def find_neighbors(self, node: ActionGraphNode) -> List[ActionGraphNode]:
        return self.graph.neighbors(node)

    def update(self, updated_node: ActionGraphNode):
        self.graph.update(updated_node)

    def add_edge(self, parent: ActionGraphNode, child: ActionGraphNode):
        self.graph.add_edge(parent, child)

    def remove_edge(self, parent: ActionGraphNode, child: ActionGraphNode):
        self.graph.remove_edge(parent, child)

    def visualize(self):
        return nx.draw_networkx(self.graph, with_labels=True)

    @property
    def nodes(self) -> Iterable:
        return self.graph.nodes()

    @property
    def edges(self) -> Iterable:
        return self.graph.edges()

    def save(self):
        # TODO 🟡: Add functionality to save the graph
        pass


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
        self.client.add_node(node)
        self._search_parents_for(node)

    def _search_parents_for(self, node: ActionGraphNode) -> None:
        input_ids = []
        parents = set()
        if node.action.remote_self:
            input_ids.append(node.action.remote_self)
        input_ids.extend(node.action.args)
        input_ids.extend(node.action.kwargs.values())
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
