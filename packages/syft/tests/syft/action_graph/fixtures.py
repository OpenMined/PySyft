# third party
import pytest

# syft absolute
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_graph import InMemoryActionGraphStore
from syft.service.action.action_graph import InMemoryGraphConfig
from syft.service.action.action_graph import NetworkXBackingStore
from syft.service.action.action_graph import NodeActionData
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject


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
def simple_in_memory_action_graph(
    in_mem_graph_config: InMemoryGraphConfig,
    verify_key: SyftVerifyKey,
) -> InMemoryActionGraphStore:
    """
    Create a simple in memory graph with 3 nodes without node mutation
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

    return graph_store


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
