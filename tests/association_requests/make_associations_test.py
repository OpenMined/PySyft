# stdlib
from typing import Dict
from typing import Tuple

# third party
import pytest

# syft absolute
import syft as sy

NETWORK_PORT = 9081
DOMAIN1_PORT = 9082
DOMAIN2_PORT = 9083
NETWORK_NODE_NAME = "test_network_1"
NETWORK_PUBLIC_HOST = f"localhost:{NETWORK_PORT}"


def get_vpn_ips(vpn_status: Dict[str, str]) -> Tuple[str, str]:
    DOMAIN_VPN_IP = vpn_status["host"]["ip"]
    for peer in vpn_status["peers"]:
        if peer["hostname"] == NETWORK_NODE_NAME:
            NETWORK_VPN_IP = peer["ip"]

    return DOMAIN_VPN_IP, NETWORK_VPN_IP


@pytest.mark.integration
def test_domain1_association_network1() -> None:
    domain = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    # get the vpn ips
    vpn_status = domain.vpn_status()
    DOMAIN_VPN_IP, NETWORK_VPN_IP = get_vpn_ips(vpn_status=vpn_status)

    domain.apply_to_network(
        domain_vpn_ip=DOMAIN_VPN_IP,
        network_vpn_ip=NETWORK_VPN_IP,
    )

    network = sy.login(
        email="info@openmined.org", password="changethis", port=NETWORK_PORT
    )
    associations = network.association.all()
    for association in associations:
        if association["node_address"] == domain.target_id.id.no_dash:
            request_id = int(association["association_id"])

    network.association[request_id].accept()
    assert domain.association.all()[0]["status"] == "ACCEPTED"


@pytest.mark.integration
def test_domain2_association_network1() -> None:
    domain = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN2_PORT
    )

    # get the vpn ips
    vpn_status = domain.vpn_status()
    DOMAIN_VPN_IP, NETWORK_VPN_IP = get_vpn_ips(vpn_status=vpn_status)

    domain.apply_to_network(
        domain_vpn_ip=DOMAIN_VPN_IP,
        network_vpn_ip=NETWORK_VPN_IP,
    )

    network = sy.login(
        email="info@openmined.org", password="changethis", port=NETWORK_PORT
    )
    associations = network.association.all()
    for association in associations:
        if association["node_address"] == domain.target_id.id.no_dash:
            request_id = int(association["association_id"])

    network.association[request_id].accept()
    assert domain.association.all()[0]["status"] == "ACCEPTED"
