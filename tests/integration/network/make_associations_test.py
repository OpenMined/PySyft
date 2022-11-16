# stdlib
import os
import time

# third party
import pytest

# syft absolute
import syft as sy

NETWORK_PORT = 9081
DOMAIN1_PORT = 9082
DOMAIN2_PORT = 9083

HOST_IP = os.environ.get("HOST_IP", "localhost")


@pytest.mark.network
def test_domain1_association_network1() -> None:
    network_guest = sy.login(port=NETWORK_PORT)

    domain = sy.login(
        email="info@openmined.org",
        password="changethis",
        port=DOMAIN1_PORT,
        url=HOST_IP,
    )

    domain.apply_to_network(client=network_guest)

    time.sleep(5)

    network = sy.login(
        email="info@openmined.org",
        password="changethis",
        port=NETWORK_PORT,
        url=HOST_IP,
    )
    associations = network.association.all()
    for association in associations:
        if association["node_address"] == domain.target_id.id.no_dash:
            request_id = int(association["association_id"])

    network.association[request_id].accept()

    time.sleep(5)

    assert domain.association.all()[0]["status"] == "ACCEPTED"
