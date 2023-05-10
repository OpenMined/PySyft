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
from result import Err

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
from syft.service.action.action_graph_service import ExecutionStatusPartitionKey
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject
from syft.store.document_store import QueryKeys
from syft.store.locks import NoLockingConfig
from syft.types.datetime import DateTime
from syft.types.syft_metaclass import Empty

# relative
from .fixtures import create_action_node
from .fixtures import create_action_obj_node


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


def test_node_action_data_update() -> None:
    node_action_data_update = NodeActionDataUpdate()

    assert node_action_data_update.id == Empty
    assert node_action_data_update.type == Empty
    assert node_action_data_update.status == Empty
    assert node_action_data_update.retry == Empty
    assert node_action_data_update.created_at == Empty
    assert node_action_data_update.credentials == Empty
    # only updated_at is not empty
    assert isinstance(node_action_data_update.updated_at, DateTime)
    assert len(node_action_data_update.to_dict(exclude_empty=True)) == 1
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
    update_node = NodeActionDataUpdate(
        status=ExecutionStatus.DONE, is_mutagen=True, is_mutated=True
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
    networkx_store_with_nodes: NetworkXBackingStore, verify_key: SyftVerifyKey
) -> None:
    """
    Test the save and load methods of NetworkXBackingStore to a default location.
    These functions rely on the serialization and deserialization methods of the store.
    """
    # save the store to and from the default location
    networkx_store_with_nodes.save()
    default_in_mem_graph_config = InMemoryGraphConfig()
    networkx_store_2 = NetworkXBackingStore(default_in_mem_graph_config)
    assert networkx_store_2.nodes() == networkx_store_with_nodes.nodes()
    assert networkx_store_2.edges() == networkx_store_with_nodes.edges()
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


def test_networkx_backing_store_subgraph(
    networkx_store_with_nodes: NetworkXBackingStore, verify_key: SyftVerifyKey
):
    processing_status = ExecutionStatus.PROCESSING
    qks = QueryKeys(qks=[ExecutionStatusPartitionKey.with_obj(processing_status)])
    subgraph = networkx_store_with_nodes.subgraph(qks)
    assert len(subgraph.nodes()) == 3
    assert len(subgraph.edges()) == 0
    # add a node with a status DONE
    action_node: NodeActionData = create_action_node(verify_key)
    action_node.status = ExecutionStatus.DONE
    networkx_store_with_nodes.set(uid=action_node.id, data=action_node)
    done_status = ExecutionStatus.DONE
    qks2 = QueryKeys(qks=[ExecutionStatusPartitionKey.with_obj(done_status)])
    subgraph2 = networkx_store_with_nodes.subgraph(qks2)
    assert len(subgraph2.nodes()) == 1
    assert len(subgraph2.edges()) == 0


def test_in_memory_action_graph_store_init(
    in_mem_graph_config: InMemoryGraphConfig,
) -> None:
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)

    assert graph_store.store_config == in_mem_graph_config
    assert isinstance(graph_store.graph, NetworkXBackingStore)
    assert isinstance(graph_store.graph.db, nx.DiGraph)


def test_in_memory_action_graph_store_set_get_delete_no_mutations(
    in_mem_graph_store: InMemoryActionGraphStore,
    verify_key: SyftVerifyKey,
) -> None:
    """
    Test these methods of InMemoryActionGraphStore: set, get, delete, nodes, edges, is_parent
    when there is no mutations.
    """
    # add the first node
    action_obj_node: NodeActionData = create_action_obj_node(verify_key)
    result = in_mem_graph_store.set(action_obj_node, credentials=verify_key)
    assert result.ok() == action_obj_node
    assert len(in_mem_graph_store.nodes(verify_key).ok()) == 1
    assert len(in_mem_graph_store.edges(verify_key).ok()) == 0
    assert (
        in_mem_graph_store.get(action_obj_node.id, verify_key).ok() == action_obj_node
    )

    # add the second node which is the child of the first node
    action_node: NodeActionData = create_action_node(verify_key)
    result2 = in_mem_graph_store.set(
        action_node, credentials=verify_key, parent_uids=[action_obj_node.id]
    )
    assert result2.ok() == action_node
    assert len(in_mem_graph_store.nodes(verify_key).ok()) == 2
    assert len(in_mem_graph_store.edges(verify_key).ok()) == 1
    assert in_mem_graph_store.get(action_node.id, verify_key).ok() == action_node
    assert (
        in_mem_graph_store.is_parent(
            parent=action_obj_node.id, child=action_node.id
        ).ok()
        is True
    )

    # add the third node which is the child of the first and second node
    action_node_2: NodeActionData = create_action_node(verify_key)
    result3 = in_mem_graph_store.set(
        action_node_2,
        credentials=verify_key,
        parent_uids=[action_obj_node.id, action_node.id],
    )
    assert result3.ok() == action_node_2
    assert len(in_mem_graph_store.nodes(verify_key).ok()) == 3
    assert len(in_mem_graph_store.edges(verify_key).ok()) == 3
    assert in_mem_graph_store.get(action_node_2.id, verify_key).ok() == action_node_2
    assert (
        in_mem_graph_store.is_parent(
            parent=action_obj_node.id, child=action_node_2.id
        ).ok()
        is True
    )
    assert (
        in_mem_graph_store.is_parent(parent=action_node.id, child=action_node_2.id).ok()
        is True
    )
    assert (
        in_mem_graph_store.is_parent(parent=action_node_2.id, child=action_node.id).ok()
        is False
    )

    # delete the first node
    result4 = in_mem_graph_store.delete(action_obj_node.id, verify_key)
    assert result4.ok() is True
    assert len(in_mem_graph_store.nodes(verify_key).ok()) == 2
    assert len(in_mem_graph_store.edges(verify_key).ok()) == 1
    assert (
        in_mem_graph_store.is_parent(parent=action_node.id, child=action_node_2.id).ok()
        is True
    )
    # trying to get the deleted note should result in an Err
    assert isinstance(in_mem_graph_store.get(action_obj_node.id, verify_key), Err)


def test_in_memory_action_graph_store_update(
    in_mem_graph_store: InMemoryActionGraphStore,
    verify_key: SyftVerifyKey,
) -> None:
    action_obj_node: NodeActionData = create_action_obj_node(verify_key)
    result = in_mem_graph_store.set(action_obj_node, credentials=verify_key).ok()
    update_node = NodeActionDataUpdate(
        status=ExecutionStatus.DONE, is_mutagen=True, is_mutated=True
    )
    result2 = in_mem_graph_store.update(
        uid=result.id, data=update_node, credentials=verify_key
    ).ok()
    assert result2.id == result.id
    assert in_mem_graph_store.get(result.id, verify_key).ok() == result2
    assert result2.status == ExecutionStatus.DONE
    assert result2.is_mutagen is True
    assert result2.is_mutated is True
    assert isinstance(result2.updated_at, DateTime)


def test_simple_in_memory_action_graph(
    simple_in_memory_action_graph: InMemoryActionGraphStore,
    verify_key: SyftVerifyKey,
) -> None:
    """
    node_1: action_obj_node_a
    node_2: action_obj_node_b
    node_3: action -> a + b = c
    """
    assert len(simple_in_memory_action_graph.edges(verify_key).ok()) == 2
    assert len(simple_in_memory_action_graph.nodes(verify_key).ok()) == 3
    # the nodes should be in the order of how they were added
    nodes = list(simple_in_memory_action_graph.nodes(verify_key).ok())
    node_1: NodeActionData = nodes[0][1]["data"]
    node_2: NodeActionData = nodes[1][1]["data"]
    node_3: NodeActionData = nodes[2][1]["data"]
    assert (
        simple_in_memory_action_graph.is_parent(parent=node_1.id, child=node_3.id).ok()
        is True
    )
    assert (
        simple_in_memory_action_graph.is_parent(parent=node_2.id, child=node_3.id).ok()
        is True
    )
    assert (
        simple_in_memory_action_graph.is_parent(parent=node_3.id, child=node_1.id).ok()
        is False
    )


def test_simple_in_memory_action_graph_query(
    simple_in_memory_action_graph: InMemoryActionGraphStore,
    verify_key: SyftVerifyKey,
) -> None:
    qks = QueryKeys(
        qks=[ExecutionStatusPartitionKey.with_obj(ExecutionStatus.PROCESSING)]
    )
    result = simple_in_memory_action_graph.query(qks, verify_key).ok()
    # the nodes should be in the order of how they were added
    nodes = list(simple_in_memory_action_graph.nodes(verify_key).ok())
    node_1: NodeActionData = nodes[0][1]["data"]
    node_2: NodeActionData = nodes[1][1]["data"]
    node_3: NodeActionData = nodes[2][1]["data"]
    assert result[0] == node_1.id
    assert result[1] == node_2.id
    assert result[2] == node_3.id
    # change the status of a node and do the query again
    node_1.status = ExecutionStatus.DONE
    done_qks = QueryKeys(
        qks=[ExecutionStatusPartitionKey.with_obj(ExecutionStatus.DONE)]
    )
    done_result = simple_in_memory_action_graph.query(done_qks, verify_key).ok()
    processing_result = simple_in_memory_action_graph.query(qks, verify_key).ok()
    assert done_result[0] == node_1.id
    assert processing_result[0] == node_2.id
    assert processing_result[1] == node_3.id
