"""
Tests for the ActionGraphService in /syft/src/syft/service/action/action_graph_service.py
"""
# third party
import networkx as nx

# syft absolute
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.node.worker import Worker
from syft.service.action.action_graph import ExecutionStatus
from syft.service.action.action_graph import InMemoryActionGraphStore
from syft.service.action.action_graph import NetworkXBackingStore
from syft.service.action.action_graph import NodeActionData
from syft.service.action.action_graph import NodeActionDataUpdate
from syft.service.action.action_graph import NodeType
from syft.service.action.action_graph_service import ActionGraphService
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject
from syft.service.context import AuthedServiceContext
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.types.datetime import DateTime
from syft.types.uid import UID


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
        node_1: action_obj_a = [1,2,3]
        node_2: action_obj_b = [2,3,4]
        node_3: action -> add(a, b)
        node_4: action_obj = a + b  (automatically created)
    """
    action_obj_a = ActionObject.from_obj([1, 2, 3])
    action_obj_b = ActionObject.from_obj([2, 3, 4])
    in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj_a
    )
    in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj_b
    )
    action = Action(
        path="action.execute",
        op="__add__",
        remote_self=action_obj_a.syft_lineage_id,
        args=[action_obj_b.syft_lineage_id],
        kwargs={},
    )
    action_node, result_node = in_mem_action_graph_service.add_action(
        context=authed_context, action=action
    )

    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 4
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 3

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
    Test the `add_action` method of ActionGraphService when mutation occurs.
    Scenario: We first create a np array, change its type, then change a value
              at a specific index, then do an addition on the mutated value.
              The final graph has 11 nodes, 10 edges, 2 mutagen nodes and 2 mutated nodes
        node_1: action_obj_d = [1,2,3]
        node_2: action -> np.array(d)
        node_3: action_obj = np.array([1,2,3])  (automatically created)
        node_4: as_type_action_obj = 'np.int32'
        node_5: action -> d = np.astype(d, 'np.int32')  (first mutation)
        node_6: idx_action_obj = 2
        node_7: item_val_action_obj = 5
        node_8: action -> d[2] = 5  (second mutation)
        node_9: action_obj_e = 48
        node_10: action -> d + e
        node_11: action_obj_f = d + 48  (automatically created)
    """
    # node_1: action_obj_d = [1,2,3]
    action_obj_d = ActionObject.from_obj([1, 2, 3])
    in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj_d
    )
    # node_2: action -> np.array(d)
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
    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 3
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 2
    assert action_node.id == action.id
    assert result_node.id == action.result_id.id
    assert action_node.type == NodeType.ACTION
    assert result_node.type == NodeType.ACTION_OBJECT
    assert result_node.updated_at is None
    assert result_node.is_mutated is False
    assert result_node.is_mutagen is False
    assert result_node.next_mutagen_node is None
    assert result_node.last_nm_mutagen_node is None
    # node_3 is the result_node that's automatically created
    # node_4: as_type_action_obj = 'np.int32'
    as_type_action_obj = ActionObject.from_obj("np.int32")
    in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=as_type_action_obj
    )
    # node_5: action -> d = np.astype(d, 'np.int32') -- mutation occurs
    action2 = Action(
        path="action.execute",
        op="astype",
        remote_self=action.result_id,
        args=[as_type_action_obj.syft_lineage_id],
        kwargs={},
        result_id=action.result_id,
    )
    action_node_2, result_node_2 = in_mem_action_graph_service.add_action(
        context=authed_context, action=action2
    )
    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 5
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 4
    assert action_node_2.type == NodeType.ACTION
    assert result_node_2.type == NodeType.ACTION_OBJECT
    assert result_node_2 == result_node
    assert action_node_2.is_mutagen is True
    assert action_node_2.is_mutated is False
    assert result_node_2.is_mutated is True
    assert result_node_2.is_mutagen is False
    assert result_node_2.next_mutagen_node == action_node_2.id
    assert result_node_2.last_nm_mutagen_node == action_node_2.id
    assert action_node_2.next_mutagen_node is None
    assert action_node_2.last_nm_mutagen_node is None
    # node_6: idx_action_obj = 2
    idx_action_obj = ActionObject.from_obj(2)
    in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=idx_action_obj
    )
    # node_7: item_val_action_obj = 5
    item_val_action_obj = ActionObject.from_obj(5)
    in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=item_val_action_obj
    )
    # node_8: action -> d[2] = 5  (second mutagen node)
    action3 = Action(
        path="action.execute",
        op="__setitem__",
        remote_self=action.result_id,
        args=[idx_action_obj.syft_lineage_id, item_val_action_obj.syft_lineage_id],
        kwargs={},
        result_id=action.result_id,
    )
    action_node_3, result_node_3 = in_mem_action_graph_service.add_action(
        context=authed_context, action=action3
    )
    assert action.result_id == action2.result_id == action3.result_id
    assert result_node_3 == action_node_2
    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 8
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 7
    # the action_node_3 is the last non-mutated mutagen node in the chain
    assert action_node_3.is_mutagen is True
    assert action_node_3.is_mutated is False
    assert action_node_3.next_mutagen_node is None
    assert action_node_3.last_nm_mutagen_node is None
    # action_node_2 should be changed accordingly
    assert action_node_2.is_mutagen is True
    assert action_node_2.is_mutated is True
    assert action_node_2.next_mutagen_node == action_node_3.id
    assert action_node_2.last_nm_mutagen_node == action_node_3.id
    # result_node should be changed accordingly
    assert result_node.is_mutagen is False
    assert result_node.is_mutated is True
    assert result_node.next_mutagen_node == action_node_2.id
    assert result_node.last_nm_mutagen_node == action_node_3.id
    # node_9: action_obj_e = 48
    action_obj_e = ActionObject.from_obj(48)
    in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj_e
    )
    # node_10: action -> d + e
    action4 = Action(
        path="action.execute",
        op="__add__",
        remote_self=action.result_id,
        args=[action_obj_e.syft_lineage_id],
        kwargs={},
    )
    action_node_4, result_node_4 = in_mem_action_graph_service.add_action(
        context=authed_context, action=action4
    )
    # node_11: action_obj_f = d + 48  (= the result_node_4 that's automatically created)
    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 11
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 10
    # the __add__ node (action_node_4) should be a
    # direct child of the __setitem__ (action_node_3) node
    assert (
        in_mem_action_graph_service.store.is_parent(
            parent=action_node_3.id, child=action_node_4.id
        ).ok()
        is True
    )
    # action_node_4 and result_node_4 do not belong to the mutation chain
    assert action_node_4.is_mutagen is False
    assert action_node_4.is_mutated is False
    assert action_node_4.next_mutagen_node is None
    assert action_node_4.last_nm_mutagen_node is None
    # result_node should be changed accordingly
    assert result_node_4.is_mutagen is False
    assert result_node_4.is_mutated is False
    assert result_node_4.next_mutagen_node is None
    assert result_node_4.last_nm_mutagen_node is None


def test_action_graph_service_get_remove_nodes(
    in_mem_action_graph_service: ActionGraphService,
    authed_context: AuthedServiceContext,
) -> None:
    """
    Test the get and remove_node method of the ActionGraphService
    """
    action_obj_a = ActionObject.from_obj([1, 2, 3])
    action_obj_b = ActionObject.from_obj([2, 3, 4])
    action_obj_node_a: NodeActionData = in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj_a
    )
    action_obj_node_b: NodeActionData = in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj_b
    )
    action = Action(
        path="action.execute",
        op="__add__",
        remote_self=action_obj_a.syft_lineage_id,
        args=[action_obj_b.syft_lineage_id],
        kwargs={},
    )
    action_node, result_node = in_mem_action_graph_service.add_action(
        context=authed_context, action=action
    )
    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 4
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 3
    nodes = set(
        dict(in_mem_action_graph_service.get_all_nodes(context=authed_context)).keys()
    )
    # test the get method
    assert action_obj_node_a == in_mem_action_graph_service.get(
        uid=action_obj_a.id, context=authed_context
    )
    assert action_obj_node_b == in_mem_action_graph_service.get(
        uid=action_obj_b.id, context=authed_context
    )
    assert action_node == in_mem_action_graph_service.get(
        uid=action.id, context=authed_context
    )
    # test the remove_node method
    removed_result: SyftSuccess = in_mem_action_graph_service.remove_node(
        authed_context, action_node.id
    )
    assert (
        removed_result.message
        == f"Successfully deleted node with uid: {action.id} from the graph."
    )
    assert len(in_mem_action_graph_service.get_all_nodes(authed_context)) == 3
    assert len(in_mem_action_graph_service.get_all_edges(authed_context)) == 0
    nodes_after_remove = set(
        dict(in_mem_action_graph_service.get_all_nodes(context=authed_context)).keys()
    )
    assert action_node.id == (nodes - nodes_after_remove).pop()


def test_action_graph_service_update(
    in_mem_action_graph_service: ActionGraphService,
    authed_context: AuthedServiceContext,
) -> None:
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
    action_node, _ = in_mem_action_graph_service.add_action(
        context=authed_context, action=action
    )
    update_data = NodeActionDataUpdate(
        status=ExecutionStatus.DONE,
        is_mutagen=True,
        is_mutated=True,
        next_mutagen_node=UID(),
        last_nm_mutagen_node=UID(),
    )
    updated_node: NodeActionData = in_mem_action_graph_service.update(
        context=authed_context, uid=action_node.id, node_data=update_data
    )
    assert updated_node.id == action_node.id
    assert updated_node.type == NodeType.ACTION
    for k, v in update_data.to_dict(exclude_empty=True).items():
        assert getattr(updated_node, k) == v


def test_action_graph_service_status(
    in_mem_action_graph_service: ActionGraphService,
    authed_context: AuthedServiceContext,
) -> None:
    """
    Test the update_action_status and get_by_action_status methods
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
    action_node, _ = in_mem_action_graph_service.add_action(
        context=authed_context, action=action
    )

    assert (
        len(
            in_mem_action_graph_service.get_by_action_status(
                authed_context, ExecutionStatus.PROCESSING
            )
        )
        == 3
    )

    updated_node = in_mem_action_graph_service.update_action_status(
        context=authed_context, action_id=action_node.id, status=ExecutionStatus.DONE
    )

    assert updated_node.status == ExecutionStatus.DONE

    done_nodes = in_mem_action_graph_service.get_by_action_status(
        authed_context, ExecutionStatus.DONE
    )
    assert len(done_nodes) == 1
    assert done_nodes[0] == updated_node.id  # should be just updated_node?
    assert (
        len(
            in_mem_action_graph_service.get_by_action_status(
                authed_context, ExecutionStatus.PROCESSING
            )
        )
        == 2
    )


def test_action_graph_service_get_by_verify_key(
    worker: Worker,
    in_mem_action_graph_service: ActionGraphService,
) -> None:
    verify_key: SyftVerifyKey = SyftSigningKey.generate().verify_key
    verify_key_2: SyftVerifyKey = SyftSigningKey.generate().verify_key
    assert verify_key_2 != verify_key
    authed_context = AuthedServiceContext(credentials=verify_key, node=worker)
    authed_context_2 = AuthedServiceContext(credentials=verify_key_2, node=worker)
    action_obj = ActionObject.from_obj([1, 2, 3])
    action_obj_2 = ActionObject.from_obj([2, 3, 4])
    node_1 = in_mem_action_graph_service.add_action_obj(
        context=authed_context, action_obj=action_obj
    )
    node_2 = in_mem_action_graph_service.add_action_obj(
        context=authed_context_2, action_obj=action_obj_2
    )

    assert (
        in_mem_action_graph_service.get_by_verify_key(authed_context, verify_key)[0]
        == node_1.id
    )

    assert (
        in_mem_action_graph_service.get_by_verify_key(authed_context_2, verify_key_2)[0]
        == node_2.id
    )


def test_action_graph_service_init_with_node(worker: Worker) -> None:
    action_graph_service = worker.get_service("actiongraphservice")
    assert isinstance(action_graph_service, ActionGraphService)
    assert isinstance(action_graph_service.store, InMemoryActionGraphStore)
    assert isinstance(action_graph_service.store.graph, NetworkXBackingStore)
    assert isinstance(action_graph_service.store.graph.db, nx.DiGraph)
