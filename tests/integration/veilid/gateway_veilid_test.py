# third party
import pytest

# syft absolute
import syft as sy
from syft.abstract_node import NodeType
from syft.client.domain_client import DomainClient
from syft.client.gateway_client import GatewayClient
from syft.client.protocol import SyftProtocol
from syft.service.network.node_peer import NodePeer
from syft.service.network.routes import VeilidNodeRoute
from syft.service.response import SyftSuccess
from syft.service.user.user_roles import ServiceRole


def remove_existing_peers(client):
    for peer in client.api.services.network.get_all_peers():
        res = client.api.services.network.delete_peer_by_id(peer.id)
        assert isinstance(res, SyftSuccess)


@pytest.mark.veilid
def test_domain_connect_to_gateway_veilid(domain_1_port, gateway_port):
    # Revert to the guest login, when we automatically generate the dht key
    # gateway_client: GatewayClient = sy.login_as_guest(port=gateway_port)
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Remove existing peers due to the previous gateway test
    remove_existing_peers(domain_client)
    remove_existing_peers(gateway_client)

    # Generate DHT Record
    gateway_dht_res = gateway_client.api.services.veilid.generate_vld_key()
    assert isinstance(gateway_dht_res, SyftSuccess), gateway_dht_res
    domain_dht_res = domain_client.api.services.veilid.generate_vld_key()
    assert isinstance(domain_dht_res, SyftSuccess), domain_dht_res

    # Retrieve DHT Record
    domain_veilid_route = domain_client.api.services.veilid.get_veilid_route()
    assert isinstance(domain_veilid_route, VeilidNodeRoute), domain_veilid_route
    gateway_veilid_route = gateway_client.api.services.veilid.get_veilid_route()
    assert isinstance(gateway_veilid_route, VeilidNodeRoute), gateway_veilid_route

    # Connect Domain to Gateway via Veilid
    result = domain_client.connect_to_gateway(
        gateway_client, protocol=SyftProtocol.VEILID
    )
    assert isinstance(result, SyftSuccess)

    proxy_domain_client = gateway_client.peers[0]
    domain_peer = domain_client.peers[0]
    gateway_peer = gateway_client.api.services.network.get_all_peers()[0]

    # Domain Asserts
    assert len(domain_client.peers) == 1
    assert isinstance(proxy_domain_client, DomainClient)
    assert domain_peer.node_type == NodeType.GATEWAY
    assert isinstance(domain_peer, NodePeer)
    assert isinstance(domain_peer.node_routes[0], VeilidNodeRoute)
    assert domain_peer.node_routes[0].vld_key == gateway_veilid_route.vld_key
    assert domain_client.name == proxy_domain_client.name

    # Gateway Asserts
    assert len(gateway_client.peers) == 1
    assert gateway_peer.node_type == NodeType.DOMAIN
    assert isinstance(gateway_peer.node_routes[0], VeilidNodeRoute)
    assert gateway_peer.node_routes[0].vld_key == domain_veilid_route.vld_key
    assert gateway_client.name == domain_peer.name
    assert len(gateway_client.domains) == 1
    assert len(gateway_client.enclaves) == 0

    # Proxy Domain Asserts
    assert proxy_domain_client.metadata == domain_client.metadata
    assert proxy_domain_client.user_role == ServiceRole.NONE

    domain_client = domain_client.login(
        email="info@openmined.org", password="changethis"
    )
    proxy_domain_client = proxy_domain_client.login(
        email="info@openmined.org", password="changethis"
    )

    assert proxy_domain_client.logged_in_user == "info@openmined.org"
    assert proxy_domain_client.user_role == ServiceRole.ADMIN
    assert proxy_domain_client.credentials == domain_client.credentials
    assert (
        proxy_domain_client.api.endpoints.keys() == domain_client.api.endpoints.keys()
    )
