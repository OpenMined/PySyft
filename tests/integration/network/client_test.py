# third party
import pytest

# syft absolute
import syft as sy
from syft.client.datasite_client import DatasiteClient
from syft.client.gateway_client import GatewayClient

DATASITE_PORT = 9082
NETWORK_PORT = 9081


@pytest.mark.parametrize(
    "server_metadata", [(NETWORK_PORT, GatewayClient), (DATASITE_PORT, DatasiteClient)]
)
@pytest.mark.network
def test_client_type(server_metadata):
    port, client_type = server_metadata
    client = sy.login(port=port, email="info@openmined.org", password="changethis")

    assert isinstance(client, client_type)
