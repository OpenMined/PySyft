# third party
from result import Result

# relative
from ...serde.serializable import serializable
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import service_method
from .action_graph import ActionGraphStore
from .action_object import Action


@serializable()
class ActionGraphService(AbstractService):
    store: ActionGraphStore

    def __init__(self, store: ActionGraphStore):
        self.store = store

    @service_method(path="graph.add", name="add")
    def add_action(
        self, context: AuthedServiceContext, action: Action
    ) -> Result[Action, str]:
        self.store.set(credentials=context.credentials, action=action)

    # def add_action(self, action: Action) -> None:
    #     # TODO: Handle Duplication
    #     node = ActionGraphNode.from_action(action)
    #     self._search_parents_for(node)
    #     self.client.add_node(node)

    # def _search_parents_for(self, node: ActionGraphNode) -> None:
    #     input_ids = []
    #     parents = set()
    #     if node.action.remote_self:
    #         input_ids.append(node.action.remote_self)
    #     input_ids.extend(node.action.args)
    #     input_ids.extend(node.action.kwargs.values())
    #     # search for parents in the existing nodes
    #     for _node in self.client.nodes:
    #         if _node.action.result_id in input_ids:
    #             parents.add(_node)

    #     for parent in parents:
    #         self.client.add_edge(parent, node)

    # def remove_action(self, action: Action):
    #     node = ActionGraphNode.from_action(action)
    #     self.client.remove_node(node)

    # def draw_graph(self):
    #     return self.client.visualize()

    # @property
    # def nodes(self):
    #     return self.client.nodes

    # @property
    # def edges(self):
    #     return self.client.edges

    # def is_parent(self, parent: Action, child: Action) -> bool:
    #     parent_node: ActionGraphNode = ActionGraphNode.from_action(parent)
    #     child_node: ActionGraphNode = ActionGraphNode.from_action(child)
    #     return self.client.is_parent(parent_node, child_node)

    # def save(self) -> None:
    #     self.client.save()

    # def load(self) -> None:
    #     self.client.load()
