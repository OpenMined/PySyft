# third party
import numpy as np
import pytest

# syft absolute
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_graph import InMemoryActionGraphStore
from syft.service.action.action_graph import InMemoryGraphConfig
from syft.service.action.action_graph import NetworkXBackingStore
from syft.service.action.action_graph import NodeActionData
from syft.service.action.action_graph import NodeType
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject


def create_action_obj_node(verify_key: SyftVerifyKey) -> NodeActionData:
    """
    Helper function to create an action object node of a random
    array of 3 float numbers
    """
    random_data = np.random.rand(3)
    action_obj = ActionObject.from_obj(random_data)
    action_obj_node = NodeActionData.from_action_obj(
        action_obj=action_obj, credentials=verify_key
    )
    assert action_obj_node.type == NodeType.ACTION_OBJECT

    return action_obj_node


def create_action_node(verify_key: SyftVerifyKey) -> NodeActionData:
    """
    Helper function to create an action node of a random
    array of 3 float numbers
    """
    random_data = np.random.rand(3)
    action_obj = ActionObject.from_obj(random_data)
    action = Action(
        path="action.execute",
        op="np.array",
        remote_self=None,
        args=[action_obj.syft_lineage_id],
        kwargs={},
    )
    action_node = NodeActionData.from_action(action=action, credentials=verify_key)
    assert action_node.type == NodeType.ACTION
    return action_node


@pytest.fixture
def verify_key() -> SyftVerifyKey:
    signing_key = SyftSigningKey.generate()
    verify_key: SyftVerifyKey = signing_key.verify_key
    return verify_key


@pytest.fixture
def in_mem_graph_config() -> InMemoryGraphConfig:
    return InMemoryGraphConfig()


@pytest.fixture
def networkx_store(in_mem_graph_config: InMemoryGraphConfig) -> NetworkXBackingStore:
    return NetworkXBackingStore(store_config=in_mem_graph_config)


@pytest.fixture
def networkx_store_with_nodes(
    verify_key: SyftVerifyKey, networkx_store: NetworkXBackingStore
) -> NetworkXBackingStore:
    action_obj_node: NodeActionData = create_action_obj_node(verify_key)
    action_node: NodeActionData = create_action_node(verify_key)
    action_node_2: NodeActionData = create_action_node(verify_key)
    networkx_store.set(uid=action_obj_node.id, data=action_obj_node)
    networkx_store.set(uid=action_node.id, data=action_node)
    networkx_store.set(uid=action_node_2.id, data=action_node_2)

    return networkx_store


@pytest.fixture
def in_mem_graph_store(
    in_mem_graph_config: InMemoryGraphConfig,
) -> InMemoryActionGraphStore:
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)
    return graph_store


@pytest.fixture
def simple_in_memory_action_graph(
    in_mem_graph_store: InMemoryActionGraphStore,
    verify_key: SyftVerifyKey,
) -> InMemoryActionGraphStore:
    """
    Create a simple in memory graph with 3 nodes without node mutation
        action_obj_node_a
        action_obj_node_b
        action -> a + b = c
    """
    action_obj_a = ActionObject.from_obj([1, 2, 3])
    action_obj_b = ActionObject.from_obj([2, 3, 4])
    action_obj_node_a: NodeActionData = NodeActionData.from_action_obj(
        action_obj_a, verify_key
    )
    action_obj_node_b: NodeActionData = NodeActionData.from_action_obj(
        action_obj_b, verify_key
    )
    in_mem_graph_store.set(action_obj_node_a, credentials=verify_key)
    in_mem_graph_store.set(action_obj_node_b, credentials=verify_key)
    action = Action(
        path="action.execute",
        op="__add__",
        remote_self=action_obj_a.syft_lineage_id,
        args=[action_obj_b.syft_lineage_id],
        kwargs={},
    )
    in_mem_graph_store.set(
        action,
        credentials=verify_key,
        parent_uids=[action_obj_node_a.id, action_obj_node_b.id],
    )

    return in_mem_graph_store


@pytest.fixture
def complicated_in_memory_action_graph(
    in_mem_graph_config: InMemoryGraphConfig,
    verify_key: SyftVerifyKey,
) -> InMemoryActionGraphStore:
    """
    Create a rather complicated in memory graph with multiple nodes, edges and mutation
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

    return graph_store


@pytest.fixture
def mutated_in_memory_action_graph(
    in_mem_graph_config: InMemoryGraphConfig,
    verify_key: SyftVerifyKey,
) -> InMemoryActionGraphStore:
    """
    Create an in memory graph with a lot of node mutations
        action1 -> initialization of variable a
        action2 -> a.astype('int32') = b
        action3 -> b.astype('float64') = c
        action4 -> a.astype('complex128') = d
    """
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)
    # action1 -> initialization of variable a
    action_obj_a = ActionObject.from_obj([1, 2, 3])
    action1 = Action(
        path="action.execute",
        op="np.array",
        remote_self=None,
        args=[action_obj_a.syft_lineage_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action1)
    # action2 -> a.astype('int32') = b
    as_type_action_obj = ActionObject.from_obj("np.int32")
    action2 = Action(
        path="action.execute",
        op="astype",
        remote_self=action1.result_id,
        args=[as_type_action_obj.syft_lineage_id],
        kwargs={},
        result_id=action1.result_id,
    )
    graph_store.set(credentials=verify_key, action=action2)
    # action3 -> b.astype('float64') = c
    as_type_action_obj = ActionObject.from_obj("np.float64")
    action3 = Action(
        path="action.execute",
        op="astype",
        remote_self=action2.result_id,
        args=[as_type_action_obj.syft_lineage_id],
        kwargs={},
        result_id=action2.result_id,
    )
    graph_store.set(credentials=verify_key, action=action3)
    # action4 -> a.astype('complex128') = d
    as_type_action_obj = ActionObject.from_obj("np.complex128")
    action4 = Action(
        path="action.execute",
        op="astype",
        remote_self=action1.result_id,
        args=[as_type_action_obj.syft_lineage_id],
        kwargs={},
        result_id=action1.result_id,
    )
    graph_store.set(credentials=verify_key, action=action4)

    return graph_store
