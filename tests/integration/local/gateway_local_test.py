# stdlib
import os
from secrets import token_hex
import time

# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.abstract_node import NodeType
from syft.client.domain_client import DomainClient
from syft.client.enclave_client import EnclaveClient
from syft.client.gateway_client import GatewayClient
from syft.service.network.network_service import NodePeerAssociationStatus
from syft.service.network.node_peer import NodePeer
from syft.service.network.node_peer import NodePeerConnectionStatus
from syft.service.network.utils import PeerHealthCheckTask
from syft.service.request.request import Request
from syft.service.response import SyftSuccess
from syft.service.user.user_roles import ServiceRole


def _launch(
    node_type: NodeType,
    association_request_auto_approval: bool = True,
    port: int | str | None = None,
):
    return sy.orchestra.launch(
        name=token_hex(8),
        node_type=node_type,
        dev_mode=True,
        reset=True,
        local_db=True,
        association_request_auto_approval=association_request_auto_approval,
        port=port,
        background_tasks=True,
    )


@pytest.fixture
def gateway():
    node = _launch(NodeType.GATEWAY)
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture(params=[True, False])
def gateway_association_request_auto_approval(request: pytest.FixtureRequest):
    node = _launch(NodeType.GATEWAY, association_request_auto_approval=request.param)
    yield (request.param, node)
    node.python_node.cleanup()
    node.land()


@pytest.fixture
def domain():
    node = _launch(NodeType.DOMAIN)
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture
def domain_2():
    node = _launch(NodeType.DOMAIN)
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture
def enclave():
    node = _launch(NodeType.ENCLAVE)
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture
def gateway_webserver():
    node = _launch(node_type=NodeType.GATEWAY, port="auto")
    yield node
    node.land()


@pytest.fixture
def domain_webserver():
    node = _launch(NodeType.DOMAIN, port="auto")
    yield node
    node.land()


@pytest.fixture
def domain_2_webserver():
    node = _launch(NodeType.DOMAIN, port="auto")
    yield node
    node.land()


@pytest.fixture(scope="function")
def set_network_json_env_var(gateway_webserver):
    """Set the environment variable for the network registry JSON string."""
    json_string = f"""
        {{
            "2.0.0": {{
                "gateways": [
                    {{
                        "name": "{gateway_webserver.name}",
                        "host_or_ip": "localhost",
                        "protocol": "http",
                        "port": "{gateway_webserver.port}",
                        "admin_email": "support@openmined.org",
                        "website": "https://www.openmined.org/",
                        "slack": "https://slack.openmined.org/",
                        "slack_channel": "#support"
                    }}
                ]
            }}
        }}
    """
    os.environ["NETWORK_REGISTRY_JSON"] = json_string
    yield
    # Clean up the environment variable after all tests in the module have run
    del os.environ["NETWORK_REGISTRY_JSON"]


@pytest.mark.local_node
def test_create_gateway(
    set_network_json_env_var, gateway_webserver, domain_webserver, domain_2_webserver
):
    assert isinstance(sy.gateways, sy.NetworkRegistry)
    assert len(sy.gateways) == 1
    assert len(sy.gateways.all_networks) == 1
    assert sy.gateways.all_networks[0]["name"] == gateway_webserver.name
    assert len(sy.gateways.online_networks) == 1
    assert sy.gateways.online_networks[0]["name"] == gateway_webserver.name

    gateway_client: GatewayClient = gateway_webserver.login(
        email="info@openmined.org",
        password="changethis",
    )
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    domain_client: DomainClient = domain_webserver.login(
        email="info@openmined.org",
        password="changethis",
    )
    domain_client_2: DomainClient = domain_2_webserver.login(
        email="info@openmined.org",
        password="changethis",
    )
    result = domain_client.connect_to_gateway(handle=gateway_webserver)
    assert isinstance(result, SyftSuccess)
    result = domain_client_2.connect_to_gateway(handle=gateway_webserver)
    assert isinstance(result, SyftSuccess)

    time.sleep(PeerHealthCheckTask.repeat_time * 2 + 1)
    assert len(sy.domains.all_domains) == 2
    assert len(sy.domains.online_domains) == 2
    # check for peer connection status
    for peer in gateway_client.api.services.network.get_all_peers():
        assert peer.ping_status == NodePeerConnectionStatus.ACTIVE


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

    # Try again (via client approach)
    result_2 = domain_client.connect_to_gateway(via_client=gateway_client)
    assert isinstance(result_2, SyftSuccess)

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
    assert all_peers[0].node_routes[0].priority == 1


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

    # reconnect to the gateway
    result = domain_client.connect_to_gateway(via_client=gateway_client)
    assert isinstance(result, SyftSuccess)
    all_peers = gateway_client.api.services.network.get_all_peers()
    assert len(all_peers) == 1
    domain_1_routes = all_peers[0].node_routes
    assert domain_1_routes[0].priority == 1

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


@pytest.mark.local_node
@pytest.mark.parametrize(
    "gateway_association_request_auto_approval", [False], indirect=True
)
def test_repeated_association_requests_peers_health_check(
    gateway_association_request_auto_approval, domain
):
    _, gateway = gateway_association_request_auto_approval
    gateway_client: GatewayClient = gateway.login(
        email="info@openmined.org",
        password="changethis",
    )
    domain_client: DomainClient = domain.login(
        email="info@openmined.org",
        password="changethis",
    )

    result = domain_client.connect_to_gateway(handle=gateway)
    assert isinstance(result, Request)

    result = domain_client.connect_to_gateway(handle=gateway)
    assert isinstance(result, Request)

    r = gateway_client.api.services.request.get_all()[-1].approve()
    assert isinstance(r, SyftSuccess)

    result = domain_client.connect_to_gateway(handle=gateway)
    assert isinstance(result, SyftSuccess)

    # the gateway client checks that the peer is associated
    res = gateway_client.api.services.network.check_peer_association(
        peer_id=domain_client.id
    )
    assert isinstance(res, NodePeerAssociationStatus)
    assert res.value == "PEER_ASSOCIATED"
