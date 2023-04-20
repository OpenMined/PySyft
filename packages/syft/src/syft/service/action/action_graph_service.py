# stdlib
from typing import List
from typing import Union

# relative
from ...serde.serializable import serializable
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .action_graph import ActionGraphNode
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
    ) -> Union[ActionGraphNode, SyftError]:
        result = self.store.set(credentials=context.credentials, action=action)
        if result.is_err():
            SyftError(message=result.err())
        return result.ok()

    def remove_action(
        self, context: AuthedServiceContext, action: Action
    ) -> Union[SyftSuccess, SyftError]:
        result = self.store.delete(uid=action.id, credentials=context.credentials)
        if result.is_ok():
            return SyftSuccess(
                f"Succesfully delete action with uid: {action.id} from graph"
            )

        return SyftError(message=result.err())

    def get_all_nodes(self, context: AuthedServiceContext) -> Union[List, SyftError]:
        result = self.store.nodes
        if result.is_ok():
            return result.ok()

        return SyftError(message="Failed to fetch nodes from the graph")

    def get_all_edges(self, context: AuthedServiceContext) -> Union[List, SyftError]:
        result = self.store.edges
        if result.is_ok():
            return result.ok()
        return SyftError(message="Failed to fetch nodes from the graph")
