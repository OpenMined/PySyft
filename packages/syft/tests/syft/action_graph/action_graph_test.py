"""
Tests for the classes in /syft/src/syft/service/action/action_graph.py:
    - NodeActionData, NodeActionDataUpdate
    - InMemoryStoreClientConfig, InMemoryGraphConfig
    - NetworkXBackingStore
    - InMemoryActionGraphStore
"""

# stdlib
import os
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
from syft.service.action.action_graph import NodeType
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject
from syft.store.locks import NoLockingConfig
from syft.types.datetime import DateTime
from syft.types.syft_metaclass import Empty


def test_node_action_data_from_action_obj(verify_key: SyftVerifyKey) -> None:
    action_obj = ActionObject.from_obj([2, 4, 6])
    node_action_obj = NodeActionData.from_action_obj(
        action_obj=action_obj, credentials=verify_key
    )

    assert node_action_obj.id == action_obj.id
    assert node_action_obj.user_verify_key == verify_key
    assert node_action_obj.type == NodeType.ACTION_OBJECT
    assert node_action_obj.status == ExecutionStatus.PROCESSING
    assert node_action_obj.retry == 0
    assert isinstance(node_action_obj.created_at, DateTime)
    assert node_action_obj.is_mutated is False
    assert node_action_obj.is_mutagen is False
    assert node_action_obj.next_mutagen_node is None
    assert node_action_obj.last_nm_mutagen_node is None


def test_node_action_data_from_action_no_mutagen(verify_key: SyftVerifyKey) -> None:
    """
    action -> a + b
    """
    action_obj_a = ActionObject.from_obj([2, 4, 6])
    action_obj_b = ActionObject.from_obj([2, 3, 4])
    # adding 2 action objects
    action = Action(
        path="action.execute",
        op="__add__",
        remote_self=action_obj_a.syft_lineage_id,
        args=[action_obj_b.syft_lineage_id],
        kwargs={},
    )
    node_action_data = NodeActionData.from_action(action=action, credentials=verify_key)

    assert node_action_data.id == action.id
    assert node_action_data.type == NodeType.ACTION
    assert node_action_data.user_verify_key == verify_key
    assert node_action_data.status == ExecutionStatus.PROCESSING
    assert node_action_data.retry == 0
    assert isinstance(node_action_data.created_at, DateTime)
    assert node_action_data.is_mutated is False
    assert node_action_data.is_mutagen is False
    assert node_action_data.next_mutagen_node is None
    assert node_action_data.last_nm_mutagen_node is None


def test_node_action_data_from_action_mutagen(verify_key: SyftVerifyKey) -> None:
    """
    action1 -> d = numpy.arry([1, 2, 3])
    action2 -> d.astype('int32') (this is a mutagen node)
    """
    action_obj = ActionObject.from_obj([1, 2, 3])
    action1 = Action(
        path="action.execute",
        op="np.array",
        remote_self=None,
        args=[action_obj.syft_lineage_id],
        kwargs={},
    )
    node_action_data1 = NodeActionData.from_action(
        action=action1, credentials=verify_key
    )
    as_type_action_obj = ActionObject.from_obj("np.int32")
    action2 = Action(
        path="action.execute",
        op="astype",
        remote_self=action1.result_id,
        args=[as_type_action_obj.syft_lineage_id],
        kwargs={},
        result_id=action1.result_id,
    )
    node_action_data2 = NodeActionData.from_action(
        action=action2, credentials=verify_key
    )
    assert node_action_data1.id == action1.id
    assert node_action_data2.id == action2.id
    assert node_action_data1.type == NodeType.ACTION
    assert node_action_data2.type == NodeType.ACTION
    assert node_action_data1.is_mutagen is False
    assert node_action_data2.is_mutagen is True
    assert node_action_data1.next_mutagen_node is None
    assert node_action_data1.last_nm_mutagen_node is None
    assert node_action_data2.next_mutagen_node is None
    assert node_action_data2.last_nm_mutagen_node is None


def test_node_action_data_update(verify_key: SyftVerifyKey) -> None:
    node_action_data_update = NodeActionDataUpdate()

    assert node_action_data_update.id is None
    assert node_action_data_update.type == Empty
    assert node_action_data_update.status == Empty
    assert node_action_data_update.retry == Empty
    assert node_action_data_update.created_at == Empty
    assert node_action_data_update.credentials == Empty
    assert isinstance(node_action_data_update.updated_at, DateTime)
    assert len(node_action_data_update.to_dict(exclude_empty=True)) == 2
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


def test_networkx_backing_store_node_related_methods(
    networkx_store: NetworkXBackingStore, verify_key: SyftVerifyKey
) -> None:
    """
    Test the methods related to nodes of the NetworkXBackingStore:
        get(), set(), is_parent(), edges(), nodes(), delete(), update() methods
    """
    assert isinstance(networkx_store.db, nx.DiGraph)

    # set and get an action object node
    action_obj_node: NodeActionData = create_action_obj_node(verify_key)
    networkx_store.set(uid=action_obj_node.id, data=action_obj_node)
    assert len(networkx_store.nodes()) == 1
    assert networkx_store.get(uid=action_obj_node.id) == action_obj_node

    # set and get an action node
    action_node: NodeActionData = create_action_node(verify_key)
    networkx_store.set(uid=action_node.id, data=action_node)
    assert networkx_store.get(uid=action_node.id) == action_node
    assert len(networkx_store.nodes()) == 2
    assert len(networkx_store.edges()) == 0
    assert (
        networkx_store.is_parent(parent=action_obj_node.id, child=action_node.id)
        is False
    )

    # update the action node
    # TODO: if not do `id=action_node.id`, action_node's id will become None
    update_node = NodeActionDataUpdate(
        id=action_node.id, status=ExecutionStatus.DONE, is_mutagen=True, is_mutated=True
    )
    for key, val in update_node.to_dict(exclude_empty=True).items():
        setattr(action_node, key, val)
    networkx_store.update(uid=action_node.id, data=action_node)
    updated_action_node = networkx_store.get(uid=action_node.id)

    assert updated_action_node.status == ExecutionStatus.DONE
    assert updated_action_node.updated_at == update_node.updated_at
    assert updated_action_node.is_mutagen == update_node.is_mutagen
    assert updated_action_node.is_mutated == update_node.is_mutated

    # remove a node
    assert networkx_store.exists(uid=action_obj_node.id) is True
    networkx_store.delete(uid=action_obj_node.id)
    assert len(networkx_store.nodes()) == 1
    assert networkx_store.exists(uid=action_obj_node.id) is False

    # remove the remaining node
    networkx_store.delete(uid=action_node.id)
    assert len(networkx_store.nodes()) == 0


def test_networkx_backing_store_edge_related_methods(
    networkx_store: NetworkXBackingStore, verify_key: SyftVerifyKey
) -> None:
    """
    Test the add_edge, remove_edge and find_neighbors methods of NetworkXBackingStore
    """
    # create some nodes and add them to the store
    action_obj_node: NodeActionData = create_action_obj_node(verify_key)
    action_node: NodeActionData = create_action_node(verify_key)
    action_node_2: NodeActionData = create_action_node(verify_key)
    networkx_store.set(uid=action_obj_node.id, data=action_obj_node)
    networkx_store.set(uid=action_node.id, data=action_node)
    networkx_store.set(uid=action_node_2.id, data=action_node_2)
    # add the edges between them (we are making a closed circle here)
    networkx_store.add_edge(parent=action_node.id, child=action_obj_node.id)
    networkx_store.add_edge(parent=action_obj_node.id, child=action_node_2.id)
    networkx_store.add_edge(parent=action_node_2.id, child=action_node.id)

    assert len(networkx_store.edges()) == 3
    assert (
        networkx_store.is_parent(parent=action_node.id, child=action_obj_node.id)
        is True
    )
    assert (
        networkx_store.is_parent(parent=action_obj_node.id, child=action_node_2.id)
        is True
    )
    assert (
        networkx_store.is_parent(parent=action_node_2.id, child=action_node.id) is True
    )

    # remove the edges
    networkx_store.remove_edge(parent=action_node.id, child=action_obj_node.id)
    assert len(networkx_store.edges()) == 2
    networkx_store.remove_edge(parent=action_obj_node.id, child=action_node_2.id)
    assert len(networkx_store.edges()) == 1
    networkx_store.remove_edge(parent=action_node_2.id, child=action_node.id)
    assert len(networkx_store.edges()) == 0
    assert len(networkx_store.nodes()) == 3


def test_networkx_backing_store_save_load_default(
    networkx_store: NetworkXBackingStore, verify_key: SyftVerifyKey
) -> None:
    """
    Test the save and load methods of NetworkXBackingStore to a default location.
    These functions rely on the serialization and deserialization methods of the store.
    """
    # create some nodes and add them to the store
    action_obj_node: NodeActionData = create_action_obj_node(verify_key)
    action_node: NodeActionData = create_action_node(verify_key)
    action_node_2: NodeActionData = create_action_node(verify_key)
    networkx_store.set(uid=action_obj_node.id, data=action_obj_node)
    networkx_store.set(uid=action_node.id, data=action_node)
    networkx_store.set(uid=action_node_2.id, data=action_node_2)
    # save the store to and from the default location
    networkx_store.save()
    default_in_mem_graph_config = InMemoryGraphConfig()
    networkx_store_2 = NetworkXBackingStore(default_in_mem_graph_config)
    assert networkx_store_2.nodes() == networkx_store.nodes()
    assert networkx_store_2.edges() == networkx_store.edges()
    # remove the saved file
    os.remove(default_in_mem_graph_config.client_config.file_path)


def test_networkx_backing_store_save_load_custom(verify_key: SyftVerifyKey) -> None:
    # save the store to and from a custom location
    custom_client_conf = InMemoryStoreClientConfig(
        filename="custom_action_graph.bytes", path="/tmp"
    )
    custom_in_mem_graph_config = InMemoryGraphConfig()
    custom_in_mem_graph_config.client_config = custom_client_conf
    networkx_store = NetworkXBackingStore(store_config=custom_in_mem_graph_config)
    action_obj_node: NodeActionData = create_action_obj_node(verify_key)
    action_node: NodeActionData = create_action_node(verify_key)
    action_node_2: NodeActionData = create_action_node(verify_key)
    networkx_store.set(uid=action_obj_node.id, data=action_obj_node)
    networkx_store.set(uid=action_node.id, data=action_node)
    networkx_store.set(uid=action_node_2.id, data=action_node_2)
    networkx_store.save()
    # load the store from the custom location
    networkx_store_2 = NetworkXBackingStore(custom_in_mem_graph_config)
    assert networkx_store_2.nodes() == networkx_store.nodes()
    assert networkx_store_2.edges() == networkx_store.edges()
    # remove the saved file
    os.remove(custom_in_mem_graph_config.client_config.file_path)


@pytest.mark.skip
def test_in_memory_action_graph_store(in_mem_graph_config: InMemoryGraphConfig) -> None:
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)

    assert graph_store.store_config == in_mem_graph_config
    assert isinstance(graph_store.graph, NetworkXBackingStore)
    assert isinstance(graph_store.graph.db, nx.DiGraph)


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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
