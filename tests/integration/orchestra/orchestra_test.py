# stdlib
from secrets import token_hex

# third party
import pytest
import requests

# syft absolute
import syft as sy
from syft.node.node import Node


@pytest.mark.parametrize("node_type", ["domain", "gateway", "enclave"])
def test_orchestra_python_local(node_type):
    name = token_hex(8)
    node = sy.orchestra.launch(name=name, node_type=node_type, local_db=False)

    try:
        assert isinstance(node.python_node, Node)
        assert node.python_node.name == name
        assert node.python_node.node_type == node_type
        assert node.python_node.metadata.node_type == node_type
    finally:
        node.python_node.cleanup()
        node.land()


@pytest.mark.parametrize("node_type", ["domain", "gateway", "enclave"])
def test_orchestra_python_server(node_type):
    name = token_hex(8)
    node = sy.orchestra.launch(
        name=name,
        port="auto",
        node_type=node_type,
        local_db=False,
    )

    try:
        metadata = requests.get(f"http://localhost:{node.port}/api/v2/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["name"] == name
        assert metadata.json()["node_type"] == node_type
    finally:
        node.land()
