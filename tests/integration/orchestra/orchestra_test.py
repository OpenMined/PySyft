# stdlib
from secrets import token_hex

# third party
import pytest
import requests

# syft absolute
import syft as sy
from syft.server.server import Server


@pytest.mark.parametrize("server_type", ["datasite", "gateway", "enclave"])
def test_orchestra_python_local(server_type):
    name = token_hex(8)
    server = sy.orchestra.launch(name=name, server_type=server_type)

    try:
        assert isinstance(server.python_server, Server)
        assert server.python_server.name == name
        assert server.python_server.server_type == server_type
        assert server.python_server.metadata.server_type == server_type
    finally:
        server.python_server.cleanup()
        server.land()


@pytest.mark.parametrize("server_type", ["datasite", "gateway", "enclave"])
def test_orchestra_python_server(server_type):
    name = token_hex(8)
    server = sy.orchestra.launch(
        name=name,
        port="auto",
        server_type=server_type,
    )

    try:
        metadata = requests.get(f"http://localhost:{server.port}/api/v2/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["name"] == name
        assert metadata.json()["server_type"] == server_type
    finally:
        server.land()
