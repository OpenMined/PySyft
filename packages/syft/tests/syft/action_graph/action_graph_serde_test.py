# stdlib

# third party
import pytest

# syft absolute
import syft as sy
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_graph import InMemoryActionGraphStore
from syft.service.action.action_graph import InMemoryGraphConfig
from syft.service.action.action_graph import NodeActionData
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject


def test_in_memory_action_graph_store_serde_simple(
    in_mem_graph_config: InMemoryGraphConfig,
    verify_key: SyftVerifyKey,
) -> None:
    """
    Create an InMemoryActionGraphStore, add some nodes and edges to it
    serialize it, deserialize it, then check if the deserialized
    InMemoryActionGraphStore object is equal to the original one.
    Scenario (no node mutation / updating):
        action1 -> a + b = c
        action2 -> initialization of variable d
        action3 -> c * d
    """
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)
    # create some actions and add them to the graph store
    action_obj_a = ActionObject.from_obj([2, 4, 6])
    action_obj_b = ActionObject.from_obj([2, 3, 4])
    # action1 -> a + b = c
    action1 = Action(
        path="action.execute",
        op="__add__",
        remote_self=action_obj_a.syft_lineage_id,
        args=[action_obj_b.syft_lineage_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action1)
    # action2 -> initialization of variable d
    action_obj_d = ActionObject.from_obj([1, 2, 3])
    action2 = Action(
        path="action.execute",
        op="np.array",
        remote_self=None,
        args=[action_obj_d.syft_lineage_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action2)
    # action3 -> c * d
    action3 = Action(
        path="action.execute",
        op="__mul__",
        remote_self=action1.result_id,
        args=[action2.result_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action3)

    # serializing and deserializing the graph store
    serialized_graph: bytes = sy.serialize(graph_store, to_bytes=True)
    deserialized_graph_store = sy.deserialize(serialized_graph, from_bytes=True)

    assert isinstance(deserialized_graph_store, type(graph_store))
    assert isinstance(deserialized_graph_store.graph, type(graph_store.graph))
    assert isinstance(deserialized_graph_store.graph.db, type(graph_store.graph.db))
    assert deserialized_graph_store.edges == graph_store.edges
    assert deserialized_graph_store.nodes == graph_store.nodes


@pytest.mark.xfail
def test_in_memory_action_graph_store_serde_node_update(
    in_mem_graph_config: InMemoryGraphConfig,
    verify_key: SyftVerifyKey,
) -> None:
    """
    Create an InMemoryActionGraphStore, add some nodes and edges to it
    serialize it, deserialize it, then check if the deserialized
    InMemoryActionGraphStore object is equal to the original one.
    Scenario (with node mutation / updating):
        action1 -> a + b = c
        action2 -> initialization of variable d
        action3 -> c * d
        action4 -> d.astype('int32')
        action5 -> d + 48
    """
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)
    # create some actions and add them to the graph store
    action_obj_a = ActionObject.from_obj([2, 4, 6])
    action_obj_b = ActionObject.from_obj([2, 3, 4])
    # action1 -> a + b = c
    action1 = Action(
        path="action.execute",
        op="__add__",
        remote_self=action_obj_a.syft_lineage_id,
        args=[action_obj_b.syft_lineage_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action1)
    # action2 -> initialization of variable d
    action_obj_d = ActionObject.from_obj([1, 2, 3])
    action2 = Action(
        path="action.execute",
        op="np.array",
        remote_self=None,
        args=[action_obj_d.syft_lineage_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action2)
    # action3 -> c * d
    action3 = Action(
        path="action.execute",
        op="__mul__",
        remote_self=action1.result_id,
        args=[action2.result_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action3)
    # action4 -> d.astype('int32')
    as_type_action_obj = ActionObject.from_obj("np.int32")
    action4 = Action(
        path="action.execute",
        op="astype",
        remote_self=action2.result_id,
        args=[as_type_action_obj.syft_lineage_id],
        kwargs={},
        result_id=action2.result_id,
    )
    graph_store.set(credentials=verify_key, action=action4)

    # check if the node action 2 has been mutated
    node_action_data_2: NodeActionData = graph_store.get(
        uid=action2.id, credentials=verify_key
    ).ok()
    assert node_action_data_2.is_mutated is True

    # action5 -> d + 48
    arg_action_obj = ActionObject.from_obj(48)
    action5 = Action(
        path="action.execute",
        op="__add__",
        remote_self=action4.result_id,
        args=[arg_action_obj.syft_lineage_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action5)

    # serializing and deserializing the graph store
    serialized_graph: bytes = sy.serialize(graph_store, to_bytes=True)
    deserialized_graph_store = sy.deserialize(serialized_graph, from_bytes=True)

    assert isinstance(deserialized_graph_store.graph, type(graph_store.graph))
    assert isinstance(deserialized_graph_store.graph.db, type(graph_store.graph.db))
    assert deserialized_graph_store.edges == graph_store.edges
    assert deserialized_graph_store.nodes == graph_store.nodes
