"""
Tests for the ActionGraphService in /syft/src/syft/service/action/action_graph_service.py
"""
# syft absolute
from syft.service.action.action_graph import ExecutionStatus
from syft.service.action.action_graph import InMemoryActionGraphStore
from syft.service.action.action_graph import NetworkXBackingStore
from syft.service.action.action_graph import NodeActionData
from syft.service.action.action_graph import NodeType
from syft.service.action.action_graph_service import ActionGraphService
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject
from syft.service.context import AuthedServiceContext
from syft.service.response import SyftError
from syft.types.datetime import DateTime


def test_action_graph_service_init(
    in_mem_action_graph_service: ActionGraphService,
) -> None:
    assert isinstance(in_mem_action_graph_service.store, InMemoryActionGraphStore)
    assert isinstance(in_mem_action_graph_service.store.graph, NetworkXBackingStore)


def test_action_graph_service_add_action_obj(
    in_mem_action_graph_service: ActionGraphService,
    authed_context: AuthedServiceContext,
) -> None:
    action_obj = ActionObject.from_obj([1, 2, 3])
    result: NodeActionData = in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj
    )
    assert result.id == action_obj.id
    assert result.type == NodeType.ACTION_OBJECT
    assert result.status == ExecutionStatus.PROCESSING
    assert result.user_verify_key == authed_context.credentials
    assert isinstance(result.created_at, DateTime)
    assert result.retry == 0
    assert result.is_mutated is False
    assert result.is_mutagen is False
    assert result.next_mutagen_node is None
    assert result.last_nm_mutagen_node is None
    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 1
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 0
    # add again the same node. Expect to get back the error
    err: SyftError = in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj
    )
    assert "Node already exists in the graph" in err.message
    # add another action_obj node
    action_obj_2 = ActionObject.from_obj([2, 3, 4])
    result_2: NodeActionData = in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj_2
    )
    assert result_2.id == action_obj_2.id
    assert result_2.type == NodeType.ACTION_OBJECT
    assert result_2.user_verify_key == authed_context.credentials
    assert isinstance(result.created_at, DateTime)
    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 2
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 0


def test_action_graph_service_add_action_no_mutagen(
    in_mem_action_graph_service: ActionGraphService,
    authed_context: AuthedServiceContext,
) -> None:
    """
    Test the `add_action` method of ActionGraphService when there is no
    mutagen, i.e. a node that causes mutation. Scenario:
        node_1: action_obj_d = [1,2,3]
        node_2: action -> np.array(d)
        node_3: action_obj = np.array([1,2,3])  (automatically created)
    """
    action_obj_d = ActionObject.from_obj([1, 2, 3])
    in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj_d
    )
    action = Action(
        path="action.execute",
        op="np.array",
        remote_self=None,
        args=[action_obj_d.syft_lineage_id],
        kwargs={},
    )
    action_node, result_node = in_mem_action_graph_service.add_action(
        context=authed_context, action=action
    )
    assert action_node.id == action.id
    assert action_node.type == NodeType.ACTION
    assert action_node.status == ExecutionStatus.PROCESSING
    assert action_node.retry == 0
    assert isinstance(action_node.created_at, DateTime)
    assert action_node.updated_at is None
    assert action_node.user_verify_key == authed_context.credentials
    assert action_node.is_mutated is False
    assert action_node.is_mutagen is False
    assert action_node.next_mutagen_node is None
    assert action_node.last_nm_mutagen_node is None

    assert result_node.id == action.result_id.id
    assert result_node.type == NodeType.ACTION_OBJECT
    assert result_node.status == ExecutionStatus.PROCESSING
    assert result_node.retry == 0
    assert isinstance(result_node.created_at, DateTime)
    assert result_node.updated_at is None
    assert result_node.user_verify_key == authed_context.credentials
    assert result_node.is_mutated is False
    assert result_node.is_mutagen is False
    assert result_node.next_mutagen_node is None
    assert result_node.last_nm_mutagen_node is None


def test_action_graph_service_add_action_mutagen(
    in_mem_action_graph_service: ActionGraphService,
    authed_context: AuthedServiceContext,
) -> None:
    """
    Test the `add_action` method of ActionGraphService when mutagen occurs.
    Scenario:
        node_1: action_obj_d = [1,2,3]
        node_2: action -> np.array(d)
        node_3: action_obj = np.array([1,2,3])  (automatically created)
        node_4: as_type_action_obj = 'np.int32'
        node_5: action -> d = np.astype(d, 'np.int32')
    """
    pass
