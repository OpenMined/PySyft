# stdlib
from enum import Enum
from typing import Any
from typing import Optional

# third party
import networkx as nx
import pydantic

# relative
from .action_object import Action
from .datetime import DateTime
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftBaseObject
from .syft_object import SyftObject
from .uid import UID


class ActionStatus(Enum):
    PROCESSING = 0
    DONE = 1
    FAILED = 2


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

    def __repr__(self):
        return self._repr_debug_()


class GraphClient(SyftBaseObject):
    __canonical_name__ = "GraphClient"
    __version__ = SYFT_OBJECT_VERSION_1

    client: Any


class InMemoryGraphClient(GraphClient):
    client: nx.DiGraph = nx.DiGraph()

    # TODO 🟡: Make this abstract such the we have common
    # methods to interact with the graph


class ActionGraph:
    def __init__(self, node_uid: UID, graph: GraphClient):
        self.node_uid = node_uid
        self.graph = graph

    def add_action(self, action: Action) -> None:
        node = ActionGraphNode.from_action(action)
        self.graph.client.add_node(node)
        self._add_parents_for(node)

    def _add_parents_for(self, node: ActionGraphNode) -> None:
        input_ids = []
        parents = set()
        if node.action.remote_self:
            input_ids.append(node.action.remote_self)
        input_ids.extend(node.action.args)
        input_ids.extend(node.action.kwargs.values())
        for _node in self.graph.client.nodes:
            if (
                _node.action.result_id in input_ids
                or _node.action.remote_self in input_ids
            ):
                parents.add(_node)

        for parent in parents:
            self.graph.client.add_edge(parent, node)

    def remove_action(self, action: Action):
        node = ActionGraphNode.from_action(action)
        self.graph.client.remove_node(node)

    def visualize(self):
        # TODO: Move this to graph Client level
        return nx.draw_networkx(self.graph.client)


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
