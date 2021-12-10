# third party
import pytest

# syft absolute
import syft as sy
from tests.integration.conftest import TestNodeData

NETWORK_PORT = 9081
DOMAIN1_PORT = 9082
DOMAIN2_PORT = 9083


@pytest.mark.network
def test_domain1_association_network1(test_network: TestNodeData, test_domain_1: TestNodeData) -> None:
    network_guest = sy.login(port=NETWORK_PORT)

    domain = sy.login(
        email="info@openmined.org", password="changethis", url=test_domain_1.grid_api_url
    )

    domain.apply_to_network(client=network_guest)

    network = sy.login(
        email="info@openmined.org", password="changethis", url=test_network.grid_api_url
    )
    associations = network.association.all()
    for association in associations:
        if association["node_address"] == domain.target_id.id.no_dash:
            request_id = int(association["association_id"])

    network.association[request_id].accept()
    assert domain.association.all()[0]["status"] == "ACCEPTED"


@pytest.mark.network
def test_domain2_association_network1(test_network: TestNodeData, test_domain_2: TestNodeData) -> None:
    network_guest = sy.login(url=test_network.grid_api_url)

    domain = sy.login(
        email="info@openmined.org", password="changethis", url=test_domain_2.grid_api_url
    )

    domain.apply_to_network(client=network_guest)

    network = sy.login(
        email="info@openmined.org", password="changethis", url=test_network.grid_api_url
    )
    associations = network.association.all()
    for association in associations:
        if association["node_address"] == domain.target_id.id.no_dash:
            request_id = int(association["association_id"])

    network.association[request_id].accept()
    assert domain.association.all()[0]["status"] == "ACCEPTED"
