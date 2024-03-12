# stdlib

# third party
from pydantic import ValidationError

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import PartitionKey
from ...store.document_store import QueryKeys
from ...types.uid import UID
from ..code.user_code import UserVerifyKeyPartitionKey
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .action_graph import ActionGraphStore
from .action_graph import ExecutionStatus
from .action_graph import NodeActionData
from .action_graph import NodeActionDataUpdate
from .action_graph import NodeType
from .action_object import Action
from .action_object import ActionObject

ExecutionStatusPartitionKey = PartitionKey(key="status", type_=ExecutionStatus)


@serializable()
class ActionGraphService(AbstractService):
    store: ActionGraphStore

    def __init__(self, store: ActionGraphStore):
        self.store = store

    @service_method(path="graph.add_action", name="add_action")
    def add_action(
        self, context: AuthedServiceContext, action: Action
    ) -> tuple[NodeActionData, NodeActionData] | SyftError:
        # Create a node for the action
        input_uids, output_uid = self._extract_input_and_output_from_action(
            action=action
        )
        node = NodeActionData.from_action(
            action=action, credentials=context.credentials
        )

        result = self.store.set(
            credentials=context.credentials, node=node, parent_uids=input_uids
        )

        if result.is_err():
            return SyftError(message=result.err())

        action_node = result.ok()

        if action_node.is_mutagen:
            # updated non-mutated successor for all nodes between
            # node_id and nm_successor_id
            if action.remote_self is None:
                return SyftError(message=f"action {action}'s remote_self is None")
            result = self.store.update_non_mutated_successor(
                node_id=action.remote_self.id,
                nm_successor_id=action_node.id,
                credentials=context.credentials,
            )
        else:
            # Create a node for the result object
            node = NodeActionData(
                id=output_uid,
                user_verify_key=context.credentials,
                type=NodeType.ACTION_OBJECT,
            )

            result = self.store.set(
                credentials=context.credentials,
                node=node,
                parent_uids=[action.id],
            )

        if result.is_err():
            return SyftError(message=result.err())

        result_node = result.ok()

        return action_node, result_node

    @service_method(path="graph.add_action_obj", name="add_action_obj")
    def add_action_obj(
        self, context: AuthedServiceContext, action_obj: ActionObject
    ) -> NodeActionData | SyftError:
        node = NodeActionData.from_action_obj(
            action_obj=action_obj, credentials=context.credentials
        )
        result = self.store.set(
            credentials=context.credentials,
            node=node,
        )
        if result.is_err():
            return SyftError(message=result.err())

        return result.ok()

    def _extract_input_and_output_from_action(
        self, action: Action
    ) -> tuple[set[UID], UID | None]:
        input_uids = set()

        if action.remote_self is not None:
            input_uids.add(action.remote_self.id)

        for arg in action.args:
            input_uids.add(arg.id)

        for _, kwarg in action.kwargs.items():
            input_uids.add(kwarg.id)

        output_uid = action.result_id.id if action.result_id is not None else None

        return input_uids, output_uid

    def get(
        self, uid: UID, context: AuthedServiceContext
    ) -> NodeActionData | SyftError:
        result = self.store.get(uid=uid, credentials=context.credentials)
        if result.is_err():
            return SyftError(message=result.err())
        return result.ok()

    def remove_node(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        result = self.store.delete(
            uid=uid,
            credentials=context.credentials,
        )
        if result.is_ok():
            return SyftSuccess(
                message=f"Successfully deleted node with uid: {uid} from the graph."
            )

        return SyftError(message=result.err())

    def get_all_nodes(self, context: AuthedServiceContext) -> list | SyftError:
        result = self.store.nodes(context.credentials)
        if result.is_ok():
            return result.ok()

        return SyftError(message="Failed to fetch nodes from the graph")

    def get_all_edges(self, context: AuthedServiceContext) -> list | SyftError:
        result = self.store.edges(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message="Failed to fetch nodes from the graph")

    def update(
        self,
        context: AuthedServiceContext,
        uid: UID,
        node_data: NodeActionDataUpdate,
    ) -> NodeActionData | SyftError:
        result = self.store.update(
            uid=uid, data=node_data, credentials=context.credentials
        )
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def update_action_status(
        self,
        context: AuthedServiceContext,
        action_id: UID,
        status: ExecutionStatus,
    ) -> SyftSuccess | SyftError:
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

    def get_by_action_status(
        self, context: AuthedServiceContext, status: ExecutionStatus
    ) -> list[NodeActionData] | SyftError:
        qks = QueryKeys(qks=[ExecutionStatusPartitionKey.with_obj(status)])

        result = self.store.query(qks=qks, credentials=context.credentials)
        if result.is_ok():
            return result.ok()

        return SyftError(message=result.err())

    def get_by_verify_key(
        self, context: AuthedServiceContext, verify_key: SyftVerifyKey
    ) -> list[NodeActionData] | SyftError:
        # TODO: Add a Query for Credentials as well,
        qks = QueryKeys(qks=[UserVerifyKeyPartitionKey.with_obj(verify_key)])

        result = self.store.query(qks=qks, credentials=context.credentials)
        if result.is_ok():
            return result.ok()

        return SyftError(message=result.err())
