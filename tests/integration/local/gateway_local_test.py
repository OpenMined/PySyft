# stdlib
from secrets import token_hex

# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.abstract_node import NodeType
from syft.client.domain_client import DomainClient
from syft.client.enclave_client import EnclaveClient
from syft.client.gateway_client import GatewayClient
from syft.service.network.node_peer import NodePeer
from syft.service.request.request import Request
from syft.service.response import SyftSuccess
from syft.service.user.user_roles import ServiceRole


def launch(node_type: NodeType, association_request_auto_approval: bool = True):
    return sy.orchestra.launch(
        name=token_hex(8),
        node_type=node_type,
        dev_mode=True,
        reset=True,
        local_db=True,
        association_request_auto_approval=association_request_auto_approval,
    )


@pytest.fixture
def gateway():
    node = launch(NodeType.GATEWAY)
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture(params=[True, False])
def gateway_association_request_auto_approval(request: pytest.FixtureRequest):
    node = launch(NodeType.GATEWAY, association_request_auto_approval=request.param)
    yield (request.param, node)
    node.python_node.cleanup()
    node.land()


@pytest.fixture
def domain():
    node = launch(NodeType.DOMAIN)
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture
def domain_2():
    node = launch(NodeType.DOMAIN)
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture
def enclave():
    node = launch(NodeType.ENCLAVE)
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.mark.local_node
def test_create_gateway_client(gateway):
    client = gateway.client
    assert isinstance(client, GatewayClient)
    assert client.metadata.node_type == NodeType.GATEWAY.value


@pytest.mark.local_node
def test_domain_connect_to_gateway(gateway_association_request_auto_approval, domain):
    association_request_auto_approval, gateway = (
        gateway_association_request_auto_approval
    )
    gateway_client: GatewayClient = gateway.login(
        email="info@openmined.org",
        password="changethis",
    )
    domain_client: DomainClient = domain.login(
        email="info@openmined.org",
        password="changethis",
    )

    result = domain_client.connect_to_gateway(handle=gateway)

    if association_request_auto_approval:
        assert isinstance(result, SyftSuccess)
    else:
        assert isinstance(result, Request)
        r = gateway_client.api.services.request.get_all()[-1].approve()
        assert isinstance(r, SyftSuccess)

    # check priority
    all_peers = gateway_client.api.services.network.get_all_peers()
    assert all_peers[0].node_routes[0].priority == 1

    # Try via client approach
    result_2 = domain_client.connect_to_gateway(via_client=gateway_client)

    if association_request_auto_approval:
        assert isinstance(result_2, SyftSuccess)
    else:
        assert isinstance(result_2, Request)
        r = gateway_client.api.services.request.get_all()[-1].approve()
        assert isinstance(r, SyftSuccess)

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

    # check priority
    all_peers = gateway_client.api.services.network.get_all_peers()
    assert all_peers[0].node_routes[0].priority == 2


@pytest.mark.local_node
def test_domain_connect_to_gateway_routes_priority(gateway, domain, domain_2) -> None:
    """
    A test for routes' priority (PythonNodeRoute)
    TODO: Add a similar test for HTTPNodeRoute
    """
    gateway_client: GatewayClient = gateway.login(
        email="info@openmined.org",
        password="changethis",
    )
    domain_client: DomainClient = domain.login(
        email="info@openmined.org",
        password="changethis",
    )

    result = domain_client.connect_to_gateway(handle=gateway)
    assert isinstance(result, SyftSuccess)

    all_peers = gateway_client.api.services.network.get_all_peers()
    assert len(all_peers) == 1
    domain_1_routes = all_peers[0].node_routes
    assert domain_1_routes[0].priority == 1

    # reconnect to the gateway. The route's priority should be increased by 1
    result = domain_client.connect_to_gateway(via_client=gateway_client)
    assert isinstance(result, SyftSuccess)
    all_peers = gateway_client.api.services.network.get_all_peers()
    assert len(all_peers) == 1
    domain_1_routes = all_peers[0].node_routes
    assert domain_1_routes[0].priority == 2

    # another domain client connects to the gateway
    domain_client_2: DomainClient = domain_2.login(
        email="info@openmined.org",
        password="changethis",
    )
    result = domain_client_2.connect_to_gateway(handle=gateway)
    assert isinstance(result, SyftSuccess)

    all_peers = gateway_client.api.services.network.get_all_peers()
    assert len(all_peers) == 2
    for peer in all_peers:
        if peer.name == domain_client.metadata.name:
            assert peer.node_routes[0].priority == 2
        if peer.name == domain_client_2.metadata.name:
            assert peer.node_routes[0].priority == 1


@pytest.mark.local_node
def test_enclave_connect_to_gateway(faker: Faker, gateway, enclave):
    gateway_client = gateway.client
    enclave_client: EnclaveClient = enclave.client

    result = enclave_client.connect_to_gateway(handle=gateway)
    assert isinstance(result, SyftSuccess)

    # Try via client approach
    result_2 = enclave_client.connect_to_gateway(via_client=gateway_client)
    assert isinstance(result_2, SyftSuccess)

    assert len(enclave_client.peers) == 1
    assert len(gateway_client.peers) == 1

    proxy_enclave_client = gateway_client.peers[0]
    enclave_peer = enclave_client.peers[0]

    assert isinstance(proxy_enclave_client, EnclaveClient)
    assert isinstance(enclave_peer, NodePeer)

    assert gateway_client.name == enclave_peer.name
    assert enclave_client.name == proxy_enclave_client.name

    # Domain's peer is a gateway and vice-versa
    assert enclave_peer.node_type == NodeType.GATEWAY

    assert len(gateway_client.domains) == 0
    assert len(gateway_client.enclaves) == 1

    assert proxy_enclave_client.metadata == enclave_client.metadata
    assert proxy_enclave_client.user_role == ServiceRole.NONE

    # add a new user to enclave
    user_email, password = faker.email(), "password"
    enclave_client.register(
        name=faker.name(),
        email=user_email,
        password=password,
        password_verify=password,
    )

    enclave_client = enclave_client.login(email=user_email, password=password)
    proxy_enclave_client = proxy_enclave_client.login(
        email=user_email, password=password
    )

    assert proxy_enclave_client.logged_in_user == user_email
    assert proxy_enclave_client.user_role == enclave_client.user_role
    assert proxy_enclave_client.credentials == enclave_client.credentials
    assert (
        proxy_enclave_client.api.endpoints.keys() == enclave_client.api.endpoints.keys()
    )
