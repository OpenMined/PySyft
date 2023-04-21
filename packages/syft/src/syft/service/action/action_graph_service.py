# stdlib
from typing import List
from typing import Union

# third party
from pydantic.error_wrappers import ValidationError

# relative
from ...serde.serializable import serializable
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .action_graph import ActionGraphStore
from .action_graph import ActionStatus
from .action_graph import NodeActionData
from .action_graph import NodeActionDataUpdate
from .action_object import Action


@serializable()
class ActionGraphService(AbstractService):
    store: ActionGraphStore

    def __init__(self, store: ActionGraphStore):
        self.store = store

    @service_method(path="graph.add", name="add")
    def add_action(
        self, context: AuthedServiceContext, action: Action
    ) -> Union[NodeActionData, SyftError]:
        result = self.store.set(credentials=context.credentials, action=action)
        if result.is_err():
            SyftError(message=result.err())
        return result.ok()

    def remove_action(
        self, context: AuthedServiceContext, action: Action
    ) -> Union[SyftSuccess, SyftError]:
        result = self.store.delete(
            uid=action.id,
            credentials=context.credentials,
        )
        if result.is_ok():
            return SyftSuccess(
                f"Successfully delete action with uid: {action.id} from graph"
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

    def update(
        self,
        context: AuthedServiceContext,
        action_id: UID,
        node_data: NodeActionDataUpdate,
    ) -> Union[NodeActionData, SyftError]:
        result = self.store.update(
            uid=action_id, data=node_data, credentials=context.credentials
        )
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def update_action_status(
        self, context: AuthedServiceContext, action_id: UID, status: ActionStatus
    ) -> Union[SyftSuccess, SyftError]:
        try:
            node_data = NodeActionDataUpdate(status=status)
        except ValidationError as e:
            return SyftError(message=f"ValidationError: {e}")
        result = self.store.update(
            uid=action_id, data=node_data, credentials=context.credentials
        )
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())
