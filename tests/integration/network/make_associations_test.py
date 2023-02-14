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


def attempt_to_connect(network: sy.Network, domain: sy.Domain) -> str:
    domain.apply_to_network(client=network)

    time.sleep(5)

    associations = network.association.all()
    for association in associations:
        if sy.__version__ == "0.7.0":
            node_uid = domain.id
        else:
            node_uid = domain.node_uid
        if association["node_address"] == node_uid.no_dash:
            request_id = int(association["association_id"])

    network.association[request_id].accept()

    time.sleep(5)

    return domain.association.all()[0]["status"]


@pytest.mark.network
def test_domain1_association_network1() -> None:
    network = sy.login(
        email="info@openmined.org",
        password="changethis",
        port=NETWORK_PORT,
        url=HOST_IP,
    )

    domain = sy.login(
        email="info@openmined.org",
        password="changethis",
        port=DOMAIN1_PORT,
        url=HOST_IP,
    )

    retry_time = 3
    while retry_time > 0:
        print(f"test_domain1_association_network1 attempt: {retry_time}")
        retry_time -= 1

        try:
            status = attempt_to_connect(
                network=network,
                domain=domain,
            )
            if status == "ACCEPTED":
                break
            else:
                time.sleep(10)
        except Exception as e:
            print(f"attempt_to_connect failed. {e}")

    assert domain.association.all()[0]["status"] == "ACCEPTED"
