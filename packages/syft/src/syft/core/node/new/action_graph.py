# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
import networkx as nx

# relative
from .action_object import Action
from .action_object import ActionObject
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .uid import UID


class ActionGraphObject(SyftObject):
    __canonical_name__ = "ActionGraphObject"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    action: Optional[Action]
    action_obj: Optional[ActionObject]
    is_action: bool = True

    @staticmethod
    def from_action(action: Action):
        return ActionGraphObject(id=action.id, action=action, is_action=True)

    @staticmethod
    def from_action_obj(action_obj: ActionObject):
        return ActionGraphObject(id=action_obj.id, action_obj=action_obj)

    def __repr__(self):
        return self._repr_debug_()


class ActionGraph:
    node_uid: UID

    def __init__(self, node_uid: UID):
        self._graph = nx.DiGraph(name=node_uid)
        self.node_uid = node_uid

    def add(self, node: ActionGraphObject) -> None:
        self._graph.add_node(node.id, data=node)

    def add_relationship(
        self, nodeA: ActionGraphObject, nodeB: ActionGraphObject
    ) -> None:
        self._graph.add_edge(nodeA, nodeB)

    def remove_node(self, node: ActionGraphObject) -> None:
        self._graph.remove_node(node)

    def remove_edge(self, node: ActionGraphObject) -> None:
        self._graph.remove_edge(node)

    def neighbors_for(self, node: ActionGraphObject) -> List:
        return list(self._graph.neighbors(node))

    def visualize(self, arrows: bool = True) -> Any:
        return nx.draw_networkx(self._graph)

    def remove_all_nodes_from(self, node: ActionGraphObject):
        all_adjacent_neighbors: list = []

        def find_adjacent_neighbors(node: ActionGraphObject, neighbors: set):
            if not self._graph.neighbors(node):
                return

            my_neighbors = self._graph.neighbors(node)
            for n in my_neighbors:
                if n not in neighbors:
                    neighbors.add(n)
                    self.find_adjacent_neighbors(n, neighbors)

        find_adjacent_neighbors(node, all_adjacent_neighbors)

        self._graph.remove_nodes_from(all_adjacent_neighbors)
