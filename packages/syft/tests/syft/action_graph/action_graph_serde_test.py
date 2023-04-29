# stdlib
from typing import Any

# third party
import pytest
from pytest import FixtureRequest

# syft absolute
import syft as sy
from syft.service.action.action_graph import InMemoryActionGraphStore


@pytest.mark.parametrize(
    "obj",
    [
        "simple_in_memory_action_graph",
        "complicated_in_memory_action_graph",
        "mutated_in_memory_action_graph",
    ],
)
def test_in_memory_action_graph_serde(obj: Any, request: FixtureRequest) -> None:
    in_memory_graph: InMemoryActionGraphStore = request.getfixturevalue(obj)
    serialized_graph: bytes = sy.serialize(in_memory_graph, to_bytes=True)
    deserialized_graph = sy.deserialize(serialized_graph, from_bytes=True)

    assert isinstance(deserialized_graph, type(in_memory_graph))
    assert isinstance(deserialized_graph.graph, type(in_memory_graph.graph))
    assert isinstance(deserialized_graph.graph.db, type(in_memory_graph.graph.db))
    assert deserialized_graph.edges == in_memory_graph.edges
    assert deserialized_graph.nodes == in_memory_graph.nodes
