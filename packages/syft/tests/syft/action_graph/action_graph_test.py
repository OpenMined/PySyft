"""
Tests for the classes in the syft.service.action.action_graph module:
    - NodeActionData, NodeActionDataUpdate
    - InMemoryStoreClientConfig, InMemoryGraphConfig
    - NetworkXBackingStore
    - InMemoryActionGraphStore
"""

# stdlib
from pathlib import Path

# third party
import networkx as nx
import numpy as np
import pytest

# syft absolute
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_graph import ExecutionStatus
from syft.service.action.action_graph import InMemoryActionGraphStore
from syft.service.action.action_graph import InMemoryGraphConfig
from syft.service.action.action_graph import InMemoryStoreClientConfig
from syft.service.action.action_graph import NetworkXBackingStore
from syft.service.action.action_graph import NodeActionData
from syft.service.action.action_graph import NodeActionDataUpdate
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject
from syft.store.locks import NoLockingConfig
from syft.types.datetime import DateTime
from syft.types.syft_metaclass import Empty


def create_node_action_data(verify_key: SyftVerifyKey) -> NodeActionData:
    """
    Helper function to create a node in the action graph
    """
    random_data = np.random.rand(3)
    action_obj = ActionObject.from_obj(random_data)
    action = Action(
        path="action.execute",
        op="np.array",
        args=[action_obj.syft_lineage_id],
        kwargs={},
    )
    node_action_data = NodeActionData.from_action(action=action, credentials=verify_key)
    return node_action_data


def test_node_action_data(verify_key: SyftVerifyKey) -> None:
    # create a NodeActionData from an Action
    action_obj = ActionObject.from_obj([1, 2, 3])
    action = Action(
        path="action.execute",
        op="np.array",
        args=[action_obj.syft_lineage_id],
        kwargs={},
    )
    node_action_data = NodeActionData.from_action(action=action, credentials=verify_key)

    assert node_action_data.id == action.id
    assert node_action_data.action == action
    assert node_action_data.user_verify_key == verify_key
    assert node_action_data.status == ExecutionStatus.PROCESSING
    assert node_action_data.__hash__() == node_action_data.action.syft_history_hash

    node_action_data_duplicate = NodeActionData.from_action(
        action=action, credentials=verify_key
    )
    assert node_action_data_duplicate == node_action_data


def test_node_action_data_update() -> None:
    node_action_data_update = NodeActionDataUpdate()

    assert node_action_data_update.id == Empty
    assert node_action_data_update.action == Empty
    assert node_action_data_update.status == Empty
    assert node_action_data_update.retry == Empty
    assert node_action_data_update.created_at == Empty
    assert node_action_data_update.credentials == Empty
    assert isinstance(node_action_data_update.updated_at, DateTime)
    # TODO: set stuff to NodeActionDataUpdate
    # TODO: test node_action_data_update.to_dict(exclude=True)
    # TODO: test node_action_data_update.to_dict(exclude=False)


def test_in_memory_store_client_config() -> None:
    default_client_conf = InMemoryStoreClientConfig()
    assert default_client_conf.filename == "action_graph.bytes"
    assert default_client_conf.path == "/tmp"
    assert default_client_conf.file_path == Path("/tmp") / "action_graph.bytes"

    custom_client_conf = InMemoryStoreClientConfig(
        filename="custom_action_graph.bytes", path="/custom"
    )
    assert custom_client_conf.filename == "custom_action_graph.bytes"
    assert custom_client_conf.path == "/custom"
    assert custom_client_conf.file_path == Path("/custom") / "custom_action_graph.bytes"


def test_in_memory_graph_config() -> None:
    store_config = InMemoryGraphConfig()
    default_client_conf = InMemoryStoreClientConfig()
    locking_config = NoLockingConfig()

    assert store_config.client_config == default_client_conf
    assert store_config.store_type == NetworkXBackingStore
    assert store_config.locking_config == locking_config


def test_networkx_backing_store_create_set_get(
    in_mem_graph_config: InMemoryGraphConfig, verify_key: SyftVerifyKey
) -> None:
    """
    Test creating a NetworkXBackingStore, its get and set methods
    """
    backing_store = NetworkXBackingStore(store_config=in_mem_graph_config)
    assert isinstance(backing_store.db, nx.DiGraph)

    node: NodeActionData = create_node_action_data(verify_key)
    backing_store.set(uid=node.id, data=node)
    assert len(backing_store.nodes()) == 1
    assert backing_store.get(uid=node.id) == node

    node_2: NodeActionData = create_node_action_data(verify_key)
    backing_store.set(uid=node_2.id, data=node_2)
    assert backing_store.get(uid=node_2.id) == node_2
    assert len(backing_store.nodes()) == 2
    assert len(backing_store.edges()) == 0
    assert backing_store.is_parent(parent=node.id, child=node_2.id) is False


@pytest.mark.xfail
def test_networkx_backing_store_node_update(
    in_mem_graph_config: InMemoryGraphConfig, verify_key: SyftVerifyKey
) -> None:
    backing_store = NetworkXBackingStore(store_config=in_mem_graph_config)
    node: NodeActionData = create_node_action_data(verify_key)
    backing_store.set(uid=node.id, data=node)

    update_node = NodeActionDataUpdate()
    node_2 = create_node_action_data(verify_key)
    update_node.action = node_2.action
    update_node.status = node_2.status
    update_node.credentials = node_2.user_verify_key

    backing_store.update(uid=node.id, data=update_node)


def test_networkx_backing_store_add_remove_edge():
    """
    Test adding and removing edges, and also the find_neighbors method of the NetworkXBackingStore
    """
    pass


def test_in_memory_action_graph_store(in_mem_graph_config: InMemoryGraphConfig) -> None:
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)

    assert graph_store.store_config == in_mem_graph_config
    assert isinstance(graph_store.graph, NetworkXBackingStore)


# @pytest.mark.parametrize("graph_client", [InMemoryGraphClient])
# def test_edge_creation(worker, graph_client):
#     action_graph = ActionGraph(node_uid=worker.id, graph_client=graph_client)
#     assert action_graph

#     action_obj_a = ActionObject.from_obj([1, 2, 3])
#     action_obj_b = ActionObject.from_obj([2, 4, 5])

#     # First action -> np.array([1, 2, 3])
#     # Create a numpy array from action_obj_a
#     action_a = Action(
#         path="action.execute",
#         op="np.array",
#         args=[action_obj_a.syft_lineage_id],
#         kwargs={},
#     )
#     action_graph.add_action(action_a)
#     node = NodeActionData.from_action(action_a)
#     assert node in action_graph.client.graph.nodes

#     # Second action -> np.array([2, 4, 5])
#     # Create a numpy array from action_obj_b
#     action_b = Action(
#         path="action.execute",
#         op="np.array",
#         args=[action_obj_b.syft_lineage_id],
#         kwargs={},
#     )
#     action_graph.add_action(action_b)

#     node = NodeActionData.from_action(action_b)
#     assert node in action_graph.client.graph.nodes

#     # Third action -> action_obj_a + action_obj_b
#     # Add two arrays
#     action_add = Action(
#         path="action.execute",
#         op="__add__",
#         remote_self=action_a.result_id,
#         args=[action_b.result_id],
#         kwargs={},
#     )
#     action_graph.add_action(action_add)
#     node = NodeActionData.from_action(action_add)

#     assert node in action_graph.client.graph.nodes
#     assert action_graph.client.graph.number_of_nodes() == 3
#     assert action_graph.client.graph.number_of_edges() == 2


# def test_node_removal(worker, graph_client):
#     pass
