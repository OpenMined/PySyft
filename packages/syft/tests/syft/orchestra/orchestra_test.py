# third party
import pytest
import requests

# syft absolute
import syft as sy
from syft.node.node import Node


@pytest.mark.parametrize("node_type", ["domain", "gateway", "enclave"])
def test_orchestra_python_local(node_type):
    node = sy.orchestra.launch(name="test-domain", node_type=node_type)

    assert isinstance(node.python_node, Node)
    assert node.python_node.name == "test-domain"
    assert node.python_node.node_type.value == node_type
    assert node.python_node.metadata.node_type == node_type


@pytest.mark.parametrize("node_type", ["domain", "gateway", "enclave"])
def test_orchestra_python_server(node_type):
    node = sy.orchestra.launch(name="test-domain", port="auto", node_type=node_type)

    metadata = requests.get(f"http://localhost:{node.port}/api/v2/metadata")
    assert metadata.status_code == 200
    assert metadata.json()["name"] == "test-domain"
    assert metadata.json()["node_type"] == node_type
