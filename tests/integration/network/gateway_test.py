# syft absolute
import syft as sy
from syft.abstract_node import NodeType
from syft.client.domain_client import DomainClient
from syft.client.gateway_client import GatewayClient
from syft.service.network.node_peer import NodePeer
from syft.service.response import SyftSuccess
from syft.service.user.user_roles import ServiceRole


def test_domain_connect_to_gateway(domain_1_port, gateway_port):
    gateway_client: GatewayClient = sy.login_as_guest(port=gateway_port)

    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    assert len(domain_client.peers) == 1
    assert len(gateway_client.peers) == 1

    proxy_domain_client = gateway_client.peers[0]
    domain_peer = domain_client.peers[0]

    assert isinstance(proxy_domain_client, DomainClient)
    assert isinstance(domain_peer, NodePeer)

    # Domain's peer is a gateway and vice-versa
    assert domain_peer.node_type == NodeType.GATEWAY

    assert gateway_client.name == domain_peer.name
    assert domain_client.name == proxy_domain_client.name

    assert len(gateway_client.domains) == 1
    assert len(gateway_client.enclaves) == 0

    assert proxy_domain_client.metadata == domain_client.metadata
    assert proxy_domain_client.user_role == ServiceRole.NONE

    domain_client.login(email="info@openmined.org", password="changethis")
    proxy_domain_client.login(email="info@openmined.org", password="changethis")

    assert proxy_domain_client.logged_in_user == "info@openmined.org"
    assert proxy_domain_client.user_role == ServiceRole.ADMIN
    assert proxy_domain_client.credentials == domain_client.credentials
    assert (
        proxy_domain_client.api.endpoints.keys() == domain_client.api.endpoints.keys()
    )
