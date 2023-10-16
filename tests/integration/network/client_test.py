# third party
import pytest

# syft absolute
import syft as sy
from syft.client.domain_client import DomainClient
from syft.client.gateway_client import GatewayClient

DOMAIN_PORT = 9082
NETWORK_PORT = 9081


@pytest.mark.parametrize(
    "node_metadata", [(NETWORK_PORT, GatewayClient), (DOMAIN_PORT, DomainClient)]
)
@pytest.mark.network
def test_client_type(node_metadata):
    port, client_type = node_metadata
    client = sy.login(port=port, email="info@openmined.org", password="changethis")

    assert isinstance(client, client_type)


@pytest.mark.parametrize(
    "node_metadata", [(NETWORK_PORT, "test-gateway-1"), (DOMAIN_PORT, "test-domain-1")]
)
@pytest.mark.network
def test_client_name(node_metadata):
    port, node_name = node_metadata
    client = sy.login(port=port, email="info@openmined.org", password="changethis")

    assert client.name == node_name
