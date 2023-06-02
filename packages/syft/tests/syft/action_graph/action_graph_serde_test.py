# stdlib
from typing import Any

# third party
from networkx.classes.reportviews import NodeDataView
from networkx.classes.reportviews import OutEdgeView
import pytest
from pytest import FixtureRequest

# syft absolute
import syft as sy
from syft import Worker
from syft.client.client import SyftClient
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_graph import InMemoryActionGraphStore
from syft.service.action.action_graph import NodeActionData

# relative
from .fixtures import create_action_node


def test_node_action_data_serde(verify_key: SyftVerifyKey) -> None:
    action_node: NodeActionData = create_action_node(verify_key)
    bytes_data: bytes = sy.serialize(action_node, to_bytes=True)
    deserialized_node_action_data = sy.deserialize(bytes_data, from_bytes=True)

    assert deserialized_node_action_data == action_node


@pytest.mark.parametrize(
    "obj",
    [
        "simple_in_memory_action_graph",
    ],
)
def test_in_memory_action_graph_serde(
    obj: Any, request: FixtureRequest, verify_key: SyftVerifyKey
) -> None:
    in_memory_graph: InMemoryActionGraphStore = request.getfixturevalue(obj)
    serialized_graph: bytes = sy.serialize(in_memory_graph, to_bytes=True)
    deserialized_graph = sy.deserialize(serialized_graph, from_bytes=True)

    assert isinstance(deserialized_graph, type(in_memory_graph))
    assert isinstance(deserialized_graph.graph, type(in_memory_graph.graph))
    assert isinstance(deserialized_graph.graph.db, type(in_memory_graph.graph.db))
    assert deserialized_graph.edges(verify_key) == in_memory_graph.edges(verify_key)
    assert deserialized_graph.nodes(verify_key) == in_memory_graph.nodes(verify_key)


def test_nodes_edges_view_serde(worker: Worker) -> None:
    root_client: SyftClient = worker.root_client
    root_client.api.lib.numpy.array([1, 2, 3])
    root_client.api.lib.numpy.array([2, 3, 4])
    nodes: NodeDataView = worker.action_graph_store.nodes(root_client.credentials).ok()
    edges: OutEdgeView = worker.action_graph_store.edges(root_client.credentials).ok()
    assert sy.deserialize(sy.serialize(nodes, to_bytes=True), from_bytes=True) == nodes
    assert sy.deserialize(sy.serialize(edges, to_bytes=True), from_bytes=True) == edges
    # the client graph service's nodes and edges methods use
    # NodeDataView and OutEdgeView serialization
    assert root_client.api.services.graph.nodes() == nodes
    assert root_client.api.services.graph.edges() == edges
