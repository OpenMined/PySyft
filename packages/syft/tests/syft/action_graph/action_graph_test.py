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


def test_node_action_data_update(verify_key: SyftVerifyKey) -> None:
    node_action_data_update = NodeActionDataUpdate()

    assert node_action_data_update.id == Empty
    assert node_action_data_update.action == Empty
    assert node_action_data_update.status == Empty
    assert node_action_data_update.retry == Empty
    assert node_action_data_update.created_at == Empty
    assert node_action_data_update.credentials == Empty
    assert isinstance(node_action_data_update.updated_at, DateTime)
    assert len(node_action_data_update.to_dict(exclude_empty=True)) == 1
    assert len(node_action_data_update.to_dict(exclude_empty=False)) == len(
        vars(node_action_data_update)
    )
    assert node_action_data_update.to_dict(exclude_empty=False) == vars(
        node_action_data_update
    )

    # test when we set the attributes of NodeActionDataUpdate
    node = create_node_action_data(verify_key)
    node_action_data_update.id = node.id
    node_action_data_update.action = node.action
    node_action_data_update.status = node.status
    node_action_data_update.credentials = node.user_verify_key
    node_action_data_update.is_mutated = True

    assert node_action_data_update.id == node.id
    assert node_action_data_update.action == node.action
    assert node_action_data_update.status == node.status
    assert node_action_data_update.credentials == node.user_verify_key
    assert node_action_data_update.is_mutated is True
    assert len(node_action_data_update.to_dict(exclude_empty=True)) == 6
    assert len(node_action_data_update.to_dict(exclude_empty=True)) == 6
    assert len(node_action_data_update.to_dict(exclude_empty=False)) == len(
        vars(node_action_data_update)
    )
    assert node_action_data_update.to_dict(exclude_empty=False) == vars(
        node_action_data_update
    )


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


def test_networkx_backing_store_node_update(
    in_mem_graph_config: InMemoryGraphConfig, verify_key: SyftVerifyKey
) -> None:
    backing_store = NetworkXBackingStore(store_config=in_mem_graph_config)
    node: NodeActionData = create_node_action_data(verify_key)
    backing_store.set(uid=node.id, data=node)

    # create a new node and update the old node according to it
    update_node_data = NodeActionDataUpdate()
    node_2 = create_node_action_data(verify_key)
    update_node_data.id = node_2.id
    update_node_data.action = node_2.action
    update_node_data.status = node_2.status
    update_node_data.credentials = node_2.user_verify_key
    update_node_data.is_mutated = True

    backing_store.update(uid=node.id, data=update_node_data)

    updated_node = backing_store.get(uid=node.id)
    assert updated_node.id == update_node_data.id == node_2.id
    assert updated_node.action == update_node_data.action == node_2.action
    assert updated_node.status == update_node_data.status == node_2.status
    assert (
        updated_node.credentials
        == update_node_data.credentials
        == node_2.user_verify_key
    )
    assert updated_node.is_mutated is True
    assert updated_node.updated_at == update_node_data.updated_at


def test_in_memory_action_graph_store(in_mem_graph_config: InMemoryGraphConfig) -> None:
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)

    assert graph_store.store_config == in_mem_graph_config
    assert isinstance(graph_store.graph, NetworkXBackingStore)
    assert isinstance(graph_store.graph.db, nx.DiGraph)


def test_simple_in_memory_action_graph(
    simple_in_memory_action_graph: InMemoryActionGraphStore,
) -> None:
    """
    action1 -> a + b = c
    action2 -> initialization of variable d
    action3 -> c * d
    """
    assert len(simple_in_memory_action_graph.edges.ok()) == 2
    assert len(simple_in_memory_action_graph.nodes.ok()) == 3

    nodes = list(simple_in_memory_action_graph.nodes.ok())
    node_action_data_1: NodeActionData = nodes[0][1]["data"]
    node_action_data_2: NodeActionData = nodes[1][1]["data"]
    node_action_data_3: NodeActionData = nodes[2][1]["data"]

    assert (
        simple_in_memory_action_graph.is_parent(
            parent=node_action_data_1.id, child=node_action_data_3.id
        ).ok()
        is True
    )
    assert (
        simple_in_memory_action_graph.is_parent(
            parent=node_action_data_2.id, child=node_action_data_3.id
        ).ok()
        is True
    )
    assert (
        simple_in_memory_action_graph.is_parent(
            parent=node_action_data_1.id, child=node_action_data_2.id
        ).ok()
        is False
    )


def test_mutated_in_memory_action_graph(
    mutated_in_memory_action_graph: InMemoryActionGraphStore,
) -> None:
    """
    action1 -> initialization of variable a
    action2 -> a.astype('int32') = b
    action3 -> b.astype('float64') = c
    action4 -> a.astype('complex128') = d
    """
    assert len(mutated_in_memory_action_graph.edges.ok()) == 3
    nodes = list(mutated_in_memory_action_graph.nodes.ok())
    node_action_data_1: NodeActionData = nodes[0][1]["data"]
    node_action_data_2: NodeActionData = nodes[1][1]["data"]
    node_action_data_3: NodeActionData = nodes[2][1]["data"]
    node_action_data_4: NodeActionData = nodes[3][1]["data"]

    assert node_action_data_1.is_mutated is True
    assert node_action_data_2.is_mutated is True
    assert node_action_data_3.is_mutated is True
    assert node_action_data_4.is_mutated is False

    assert (
        mutated_in_memory_action_graph.is_parent(
            parent=node_action_data_1.id, child=node_action_data_2.id
        ).ok()
        is True
    )
    assert (
        mutated_in_memory_action_graph.is_parent(
            parent=node_action_data_2.id, child=node_action_data_3.id
        ).ok()
        is True
    )
    assert (
        mutated_in_memory_action_graph.is_parent(
            parent=node_action_data_3.id, child=node_action_data_4.id
        ).ok()
        is True
    )


def test_complicated_in_memory_action_graph(
    complicated_in_memory_action_graph: InMemoryActionGraphStore,
) -> None:
    """
    action1 -> a + b = c
    action2 -> initialization of variable d
    action3 -> c * d
    action4 -> d.astype('int32')
    action5 -> d + 48
    """
    assert len(complicated_in_memory_action_graph.edges.ok()) == 4
    assert len(complicated_in_memory_action_graph.nodes.ok()) == 5

    nodes = list(complicated_in_memory_action_graph.nodes.ok())
    node_action_data_1: NodeActionData = nodes[0][1]["data"]
    node_action_data_2: NodeActionData = nodes[1][1]["data"]
    node_action_data_3: NodeActionData = nodes[2][1]["data"]
    node_action_data_4: NodeActionData = nodes[3][1]["data"]
    node_action_data_5: NodeActionData = nodes[4][1]["data"]

    assert node_action_data_2.is_mutated is True
    assert (
        complicated_in_memory_action_graph.is_parent(
            parent=node_action_data_1.id, child=node_action_data_2.id
        ).ok()
        is False
    )
    assert (
        complicated_in_memory_action_graph.is_parent(
            parent=node_action_data_1.id, child=node_action_data_3.id
        ).ok()
        is True
    )
    assert (
        complicated_in_memory_action_graph.is_parent(
            parent=node_action_data_2.id, child=node_action_data_4.id
        ).ok()
        is True
    )
    assert (
        complicated_in_memory_action_graph.is_parent(
            parent=node_action_data_4.id, child=node_action_data_5.id
        ).ok()
        is True
    )
    assert (
        complicated_in_memory_action_graph.is_parent(
            parent=node_action_data_1.id, child=node_action_data_5.id
        ).ok()
        is False
    )


@pytest.mark.skip
def test_networkx_backing_store_add_remove_edge():
    """
    Test adding and removing edges, and also the
    find_neighbors method of the NetworkXBackingStore
    """
    pass
