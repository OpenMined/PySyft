# third party
from faker import Faker

# syft absolute
import syft as sy
from syft.abstract_node import NodeType
from syft.client.domain_client import DomainClient
from syft.client.enclave_client import EnclaveClient
from syft.client.gateway_client import GatewayClient
from syft.service.network.node_peer import NodePeer
from syft.service.response import SyftSuccess
from syft.service.user.user_roles import ServiceRole


def get_client(node_type: str):
    node = sy.orchestra.launch(
        name=sy.UID().to_string(),
        node_type=node_type,
        dev_mode=True,
        reset=True,
        local_db=True,
    )
    return node.client


def get_admin_client(node_type: str):
    node = sy.orchestra.launch(
        name=sy.UID().to_string(),
        node_type=node_type,
        dev_mode=True,
        reset=True,
        local_db=True,
    )
    return node.login(email="info@openmined.org", password="changethis")


def test_create_gateway_client(faker: Faker):
    client = get_client(NodeType.GATEWAY.value)
    assert isinstance(client, GatewayClient)
    assert client.metadata.node_type == NodeType.GATEWAY.value


def test_domain_apply_to_gateway(faker: Faker):
    gateway_client: GatewayClient = get_admin_client(NodeType.GATEWAY.value)
    domain_client: DomainClient = get_admin_client(NodeType.DOMAIN.value)

    result = domain_client.apply_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    assert len(domain_client.peers) == 1
    assert len(gateway_client.peers) == 1

    gateway_peer = gateway_client.peers[0]
    domain_peer = domain_client.peers[0]

    assert isinstance(gateway_peer, NodePeer)
    assert isinstance(domain_peer, NodePeer)

    assert gateway_client.name == domain_peer.name
    assert domain_client.name == gateway_peer.name

    proxy_domain_client = gateway_client.proxy_to(gateway_peer)
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


def test_enclave_apply_to_gateway(faker: Faker):
    gateway_client: GatewayClient = get_admin_client(NodeType.GATEWAY.value)
    enclave_client: EnclaveClient = get_admin_client(NodeType.ENCLAVE.value)

    result = enclave_client.apply_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    assert len(enclave_client.peers) == 1
    assert len(gateway_client.peers) == 1

    gateway_peer = gateway_client.peers[0]
    enclave_peer = enclave_client.peers[0]

    assert isinstance(gateway_peer, NodePeer)
    assert isinstance(enclave_peer, NodePeer)

    assert gateway_client.name == enclave_peer.name
    assert enclave_client.name == gateway_peer.name

    proxy_enclave_client = gateway_client.proxy_to(gateway_peer)
    assert proxy_enclave_client.metadata == enclave_client.metadata
    assert proxy_enclave_client.user_role == ServiceRole.NONE

    # add a new user to enclave
    user_email, password = faker.email(), "password"
    enclave_client.register(
        name=faker.name(),
        email=user_email,
        password=password,
    )

    enclave_client.login(email=user_email, password=password)
    proxy_enclave_client.login(email=user_email, password=password)

    assert proxy_enclave_client.logged_in_user == user_email
    assert proxy_enclave_client.user_role == enclave_client.user_role
    assert proxy_enclave_client.credentials == enclave_client.credentials
    assert (
        proxy_enclave_client.api.endpoints.keys() == enclave_client.api.endpoints.keys()
    )
