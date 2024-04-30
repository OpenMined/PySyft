# stdlib
import itertools
import os
from textwrap import dedent
import uuid

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.abstract_node import NodeType
from syft.client.client import HTTPConnection
from syft.client.client import SyftClient
from syft.client.domain_client import DomainClient
from syft.client.gateway_client import GatewayClient
from syft.client.gateway_client import ProxyClient
from syft.client.registry import NetworkRegistry
from syft.client.search import SearchResults
from syft.service.dataset.dataset import Dataset
from syft.service.network.association_request import AssociationRequestChange
from syft.service.network.network_service import NodePeerAssociationStatus
from syft.service.network.node_peer import NodePeer
from syft.service.network.routes import HTTPNodeRoute
from syft.service.network.routes import NodeRouteType
from syft.service.request.request import Request
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.user.user_roles import ServiceRole
from syft.types.uid import UID


@pytest.fixture(scope="function")
def set_env_var(
    gateway_port: int,
    gateway_node: str = "testgateway1",
    host_or_ip: str = "localhost",
    protocol: str = "http",
):
    """Set the environment variable for the network registry JSON string."""
    json_string = f"""
        {{
            "2.0.0": {{
                "gateways": [
                    {{
                        "name": "{gateway_node}",
                        "host_or_ip": "{host_or_ip}",
                        "protocol": "{protocol}",
                        "port": {gateway_port},
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


def _random_hash() -> str:
    return uuid.uuid4().hex[:16]


def _remove_existing_peers(client: SyftClient) -> SyftSuccess | SyftError:
    peers: list[NodePeer] | SyftError = client.api.services.network.get_all_peers()
    if isinstance(peers, SyftError):
        return peers
    for peer in peers:
        res = client.api.services.network.delete_peer_by_id(peer.id)
        if isinstance(res, SyftError):
            return res
    return SyftSuccess(message="All peers removed.")


@pytest.mark.skip(reason="Will be tested when the network registry URL works.")
def test_network_registry_from_url() -> None:
    assert isinstance(sy.gateways, NetworkRegistry)
    assert len(sy.gateways.all_networks) == len(sy.gateways.online_networks) == 1


def test_network_registry_env_var(set_env_var) -> None:
    assert isinstance(sy.gateways, NetworkRegistry)
    assert len(sy.gateways.all_networks) == len(sy.gateways.online_networks) == 1
    assert isinstance(sy.gateways[0], GatewayClient)
    assert isinstance(sy.gateways[0].connection, HTTPConnection)


def test_domain_connect_to_gateway(
    set_env_var, domain_1_port: int, gateway_port: int
) -> None:
    # check if we can see the online gateways
    assert isinstance(sy.gateways, NetworkRegistry)
    assert len(sy.gateways.all_networks) == len(sy.gateways.online_networks) == 1

    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, Request)
    assert isinstance(result.changes[0], AssociationRequestChange)

    assert len(domain_client.peers) == 1
    assert len(gateway_client.peers) == 0

    gateway_client_root = gateway_client.login(
        email="info@openmined.org", password="changethis"
    )
    res = gateway_client_root.api.services.request.get_all()[-1].approve()
    assert not isinstance(res, SyftError)

    assert len(gateway_client.peers) == 1

    # check that the domain is online on the network
    assert len(sy.domains.all_domains) == 1
    assert len(sy.domains.online_domains) == 1

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

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_dataset_search(set_env_var, gateway_port: int, domain_1_port: int) -> None:
    """
    Scenario: Connecting a domain node to a gateway node. The domain
        client then upload a dataset, which should be searchable by the syft network.
        People who install syft can see the mock data and metadata of the uploaded datasets
    """
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connect the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(sy.gateways.all_networks) == len(sy.gateways.online_networks) == 1
    assert len(sy.domains.all_domains) == len(sy.domains.online_domains) == 1

    # the domain client uploads a dataset
    input_data = np.array([1, 2, 3])
    mock_data = np.array([4, 5, 6])
    asset_name = _random_hash()
    asset = sy.Asset(name=asset_name, data=input_data, mock=mock_data)
    dataset_name = _random_hash()
    dataset = sy.Dataset(name=dataset_name, asset_list=[asset])
    dataset_res = domain_client.upload_dataset(dataset)
    assert isinstance(dataset_res, SyftSuccess)

    # test if the dataset can be searched by the syft network
    right_search = sy.search(dataset_name)
    assert isinstance(right_search, SearchResults)
    assert len(right_search) == 1
    dataset = right_search[0]
    assert isinstance(dataset, Dataset)
    assert len(dataset.assets) == 1
    assert isinstance(dataset.assets[0].mock, np.ndarray)
    assert dataset.assets[0].data is None

    # search a wrong dataset should return an empty list
    wrong_search = sy.search(_random_hash())
    assert len(wrong_search) == 0

    # the domain client delete the dataset
    domain_client.api.services.dataset.delete_by_uid(uid=dataset.id)

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_domain_gateway_user_code(
    set_env_var, domain_1_port: int, gateway_port: int
) -> None:
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # the domain client uploads a dataset
    input_data = np.array([1, 2, 3])
    mock_data = np.array([4, 5, 6])
    asset_name = _random_hash()
    asset = sy.Asset(name=asset_name, data=input_data, mock=mock_data)
    dataset_name = _random_hash()
    dataset = sy.Dataset(name=dataset_name, asset_list=[asset])
    dataset_res = domain_client.upload_dataset(dataset)
    assert isinstance(dataset_res, SyftSuccess)

    # the domain client registers a data data scientist account on its domain
    random_name: str = str(_random_hash())
    user_create_res = domain_client.register(
        name=random_name,
        email=f"{random_name}@caltech.edu",
        password="changethis",
        password_verify="changethis",
        institution="Caltech",
        website="https://www.caltech.edu/",
    )
    assert isinstance(user_create_res, SyftSuccess)

    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # the domain client connects to the gateway
    gateway_con_res = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(gateway_con_res, SyftSuccess)

    # get the proxy client to the domain, login to the data scientist account
    proxy_client = gateway_client.domains[0]
    proxy_ds = proxy_client.login(
        email=f"{random_name}@caltech.edu",
        password="changethis",
        password_verify="changethis",
    )

    # submits a request for code execution
    asset = proxy_ds.datasets[0].assets[0]

    @sy.syft_function_single_use(asset=asset)
    def mock_function(asset):
        return asset + 1

    mock_function.code = dedent(mock_function.code)
    request_res = proxy_ds.code.request_code_execution(mock_function)
    assert isinstance(request_res, Request)

    # domain client approves the request
    assert len(domain_client.requests.get_all()) == 1
    req_approve_res = domain_client.requests[-1].approve()
    assert isinstance(req_approve_res, SyftSuccess)

    # the proxy data scientist client executes the code and gets the result
    result = proxy_ds.code.mock_function(asset=asset)
    final_result = result.get()
    assert (final_result == input_data + 1).all()

    # the domain client delete the dataset
    domain_client.api.services.dataset.delete_by_uid(uid=dataset.id)

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_deleting_peers(set_env_var, domain_1_port: int, gateway_port: int) -> None:
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(domain_client.peers) == 1
    assert len(gateway_client.peers) == 1
    # check that the domain is online on the network
    assert len(sy.domains.all_domains) == 1
    assert len(sy.domains.online_domains) == 1

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)
    # check that removing peers work as expected
    assert len(sy.gateways.all_networks) == 1
    assert len(sy.domains.all_domains) == 0
    assert len(sy.domains.all_domains) == 0
    assert len(sy.domains.online_domains) == 0
    assert len(domain_client.peers) == 0
    assert len(gateway_client.peers) == 0

    # reconnect the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(domain_client.peers) == 1
    assert len(gateway_client.peers) == 1
    # check that the domain
    assert len(sy.domains.all_domains) == 1
    assert len(sy.domains.online_domains) == 1

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)
    # check that removing peers work as expected
    assert len(sy.domains.all_domains) == 0
    assert len(sy.domains.all_domains) == 0
    assert len(sy.domains.online_domains) == 0
    assert len(domain_client.peers) == 0
    assert len(gateway_client.peers) == 0


def test_add_route(set_env_var, gateway_port: int, domain_1_port: int) -> None:
    """
    Test the network service's `add_route` functionalities to add routes directly
    for a self domain.
    Scenario: Connect a domain to a gateway. The gateway adds 2 new routes to the domain
    and check their priorities.
    Then add an existed route and check if its priority gets updated.
    Check for the gateway if the proxy client to connect to the domain uses the
    route with the highest priority.
    """
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )
    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(domain_client.peers) == 1
    assert len(gateway_client.peers) == 1

    # add a new route to connect to the domain
    new_route = HTTPNodeRoute(host_or_ip="localhost", port=10000)
    domain_peer: NodePeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=domain_peer.verify_key, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(domain_peer.node_routes) == 2
    assert domain_peer.node_routes[-1].port == new_route.port

    # adding another route to the domain
    new_route2 = HTTPNodeRoute(host_or_ip="localhost", port=10001)
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=domain_peer.verify_key, route=new_route2
    )
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(domain_peer.node_routes) == 3
    assert domain_peer.node_routes[-1].port == new_route2.port
    assert domain_peer.node_routes[-1].priority == 3

    # add an existed route to the domain and check its priority gets updated
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=domain_peer.verify_key, route=domain_peer.node_routes[0]
    )
    assert "route already exists" in res.message
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(domain_peer.node_routes) == 3
    assert domain_peer.node_routes[0].priority == 4

    # the gateway gets the proxy client to the domain
    # the proxy client should use the route with the highest priority
    proxy_domain_client = gateway_client.peers[0]
    assert isinstance(proxy_domain_client, DomainClient)

    # add another existed route (port 10000)
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=domain_peer.verify_key, route=domain_peer.node_routes[1]
    )
    assert "route already exists" in res.message
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(domain_peer.node_routes) == 3
    assert domain_peer.node_routes[1].priority == 5
    # getting the proxy client using the current highest priority route should
    # give back an error since it is a route with a random port (10000)
    proxy_domain_client = gateway_client.peers[0]
    assert isinstance(proxy_domain_client, SyftError)
    assert "Failed to establish a connection with" in proxy_domain_client.message

    # the routes the domain client uses to connect to the gateway should stay the same
    gateway_peer: NodePeer = domain_client.peers[0]
    assert len(gateway_peer.node_routes) == 1

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_delete_route(set_env_var, gateway_port: int, domain_1_port: int) -> None:
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(domain_client.peers) == 1
    assert len(gateway_client.peers) == 1

    # add a new route to connect to the domain
    new_route = HTTPNodeRoute(host_or_ip="localhost", port=10000)
    domain_peer: NodePeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=domain_peer.verify_key, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(domain_peer.node_routes) == 2
    assert domain_peer.node_routes[-1].port == new_route.port

    # delete the added route
    res = gateway_client.api.services.network.delete_route(
        peer_verify_key=domain_peer.verify_key, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(domain_peer.node_routes) == 1
    assert domain_peer.node_routes[-1].port == domain_1_port

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_add_route_on_peer(set_env_var, gateway_port: int, domain_1_port: int) -> None:
    """
    Test the `add_route_on_peer` of network service.
    Connect a domain to a gateway.
    The gateway adds 2 new routes for the domain and check their priorities.
    Then add an existed route and check if its priority gets updated.
    Then the domain adds a route to itself for the gateway.
    """
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(domain_client.peers) == 1
    assert len(gateway_client.peers) == 1
    gateway_peer: NodePeer = domain_client.peers[0]
    assert len(gateway_peer.node_routes) == 1
    assert gateway_peer.node_routes[-1].priority == 1

    # adding a new route for the domain
    new_route = HTTPNodeRoute(host_or_ip="localhost", port=10000)
    domain_peer: NodePeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=domain_peer, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    gateway_peer = domain_client.peers[0]
    assert len(gateway_peer.node_routes) == 2
    assert gateway_peer.node_routes[-1].port == new_route.port
    assert gateway_peer.node_routes[-1].priority == 2

    # adding another route for the domain
    new_route2 = HTTPNodeRoute(host_or_ip="localhost", port=10001)
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=domain_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)
    gateway_peer = domain_client.peers[0]
    assert len(gateway_peer.node_routes) == 3
    assert gateway_peer.node_routes[-1].port == new_route2.port
    assert gateway_peer.node_routes[-1].priority == 3

    # add an existed route for the domain and check its priority gets updated
    existed_route = gateway_peer.node_routes[0]
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=domain_peer, route=existed_route
    )
    assert "route already exists" in res.message
    assert isinstance(res, SyftSuccess)
    gateway_peer = domain_client.peers[0]
    assert len(gateway_peer.node_routes) == 3
    assert gateway_peer.node_routes[0].priority == 4

    # the domain calls `add_route_on_peer` to to add a route to itself for the gateway
    assert len(domain_peer.node_routes) == 1
    res = domain_client.api.services.network.add_route_on_peer(
        peer=gateway_peer, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert domain_peer.node_routes[-1].port == new_route.port
    assert len(domain_peer.node_routes) == 2

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_delete_route_on_peer(
    set_env_var, gateway_port: int, domain_1_port: int
) -> None:
    """
    Connect a domain to a gateway, the gateway adds 2 new routes for the domain
    , then delete them.
    """
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    # gateway adds 2 new routes for the domain
    new_route = HTTPNodeRoute(host_or_ip="localhost", port=10000)
    new_route2 = HTTPNodeRoute(host_or_ip="localhost", port=10001)
    domain_peer: NodePeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=domain_peer, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=domain_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)

    gateway_peer: NodePeer = domain_client.peers[0]
    assert len(gateway_peer.node_routes) == 3

    # gateway delete the routes for the domain
    res = gateway_client.api.services.network.delete_route_on_peer(
        peer=domain_peer, route_id=new_route.id
    )
    assert isinstance(res, SyftSuccess)
    gateway_peer = domain_client.peers[0]
    assert len(gateway_peer.node_routes) == 2

    res = gateway_client.api.services.network.delete_route_on_peer(
        peer=domain_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)
    gateway_peer = domain_client.peers[0]
    assert len(gateway_peer.node_routes) == 1

    # gateway deletes the last the route to it for the domain
    last_route: NodeRouteType = gateway_peer.node_routes[0]
    res = gateway_client.api.services.network.delete_route_on_peer(
        peer=domain_peer, route=last_route
    )
    assert isinstance(res, SyftSuccess)
    assert "There is no routes left" in res.message
    assert len(domain_client.peers) == 0  # gateway is no longer a peer of the domain

    # The gateway client also removes the domain as a peer
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_update_route_priority(
    set_env_var, gateway_port: int, domain_1_port: int
) -> None:
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    # gateway adds 2 new routes to the domain
    new_route = HTTPNodeRoute(host_or_ip="localhost", port=10000)
    new_route2 = HTTPNodeRoute(host_or_ip="localhost", port=10001)
    domain_peer: NodePeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=domain_peer.verify_key, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=domain_peer.verify_key, route=new_route2
    )
    assert isinstance(res, SyftSuccess)

    # check if the priorities of the routes are correct
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in domain_peer.node_routes
    }
    assert routes_port_priority[domain_1_port] == 1
    assert routes_port_priority[new_route.port] == 2
    assert routes_port_priority[new_route2.port] == 3

    # update the priorities for the routes
    res = gateway_client.api.services.network.update_route_priority(
        peer_verify_key=domain_peer.verify_key, route=new_route, priority=5
    )
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in domain_peer.node_routes
    }
    assert routes_port_priority[new_route.port] == 5

    res = gateway_client.api.services.network.update_route_priority(
        peer_verify_key=domain_peer.verify_key, route=new_route2
    )
    assert isinstance(res, SyftSuccess)
    domain_peer = gateway_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in domain_peer.node_routes
    }
    assert routes_port_priority[new_route2.port] == 6

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_update_route_priority_on_peer(
    set_env_var, gateway_port: int, domain_1_port: int
) -> None:
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    # gateway adds 2 new routes for the domain to itself
    domain_peer: NodePeer = gateway_client.api.services.network.get_all_peers()[0]
    new_route = HTTPNodeRoute(host_or_ip="localhost", port=10000)
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=domain_peer, route=new_route
    )
    assert isinstance(res, SyftSuccess)

    new_route2 = HTTPNodeRoute(host_or_ip="localhost", port=10001)
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=domain_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)

    # check if the priorities of the routes are correct
    gateway_peer = domain_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in gateway_peer.node_routes
    }
    assert routes_port_priority[gateway_port] == 1
    assert routes_port_priority[new_route.port] == 2
    assert routes_port_priority[new_route2.port] == 3

    # gateway updates the route priorities for the domain remotely
    res = gateway_client.api.services.network.update_route_priority_on_peer(
        peer=domain_peer, route=new_route, priority=5
    )
    assert isinstance(res, SyftSuccess)
    res = gateway_client.api.services.network.update_route_priority_on_peer(
        peer=domain_peer, route=gateway_peer.node_routes[0]
    )
    assert isinstance(res, SyftSuccess)

    gateway_peer = domain_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in gateway_peer.node_routes
    }
    assert routes_port_priority[new_route.port] == 5
    assert routes_port_priority[gateway_port] == 6

    # Remove existing peers
    assert isinstance(_remove_existing_peers(domain_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


def test_dataset_stream(set_env_var, gateway_port: int, domain_1_port: int) -> None:
    """
    Scenario: Connecting a domain node to a gateway node. The domain
        client then upload a dataset, which should be searchable by the syft network.
        People who install syft can see the mock data and metadata of the uploaded datasets
    """
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connect the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(sy.gateways.all_networks) == len(sy.gateways.online_networks) == 1
    assert len(sy.domains.all_domains) == len(sy.domains.online_domains) == 1

    # the domain client uploads a dataset
    input_data = np.array([1, 2, 3])
    mock_data = np.array([4, 5, 6])
    asset_name = _random_hash()
    asset = sy.Asset(name=asset_name, data=input_data, mock=mock_data)
    dataset_name = _random_hash()
    dataset = sy.Dataset(name=dataset_name, asset_list=[asset])
    dataset_res = domain_client.upload_dataset(dataset)
    assert isinstance(dataset_res, SyftSuccess)

    domain_proxy_client = next(
        gateway_client.domains[i]
        for i in itertools.count()
        if gateway_client.domains[i].name == domain_client.name
    )
    root_proxy_client = domain_proxy_client.login(
        email="info@openmined.org", password="changethis"
    )
    retrieved_dataset = root_proxy_client.datasets[dataset_name]
    retrieved_asset = retrieved_dataset.assets[asset_name]
    assert np.all(retrieved_asset.data == input_data)

    # the domain client delete the dataset
    domain_client.api.services.dataset.delete_by_uid(uid=retrieved_dataset.id)


def test_peer_health_check(set_env_var, gateway_port: int, domain_1_port: int) -> None:
    """
    Scenario: Connecting a domain node to a gateway node.
    The gateway client approves the association request.
    The gateway client checks that the domain peer is associated
    TODO: check that the domain is online with `DomainRegistry.online_domains`
        Then make the domain go offline, which should be reflected when calling
        `DomainRegistry.online_domains`
    """
    # login to the domain and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # gateway checks that the domain is not yet associated
    res = gateway_client.api.services.network.check_peer_association(
        peer_id=UID(domain_client.metadata.id)
    )
    assert isinstance(res, NodePeerAssociationStatus)
    assert res.value == "PEER_NOT_FOUND"

    # connecting the domain to the gateway
    result = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(result, Request)
    assert isinstance(result.changes[0], AssociationRequestChange)

    # the gateway client approves the association request
    res = gateway_client.api.services.request.get_all()[-1].approve()
    assert not isinstance(res, SyftError)
    assert len(gateway_client.peers) == 1

    # the gateway client checks that the peer is associated
    res = gateway_client.api.services.network.check_peer_association(
        peer_id=gateway_client.peers[0].id
    )
    assert isinstance(res, NodePeerAssociationStatus)
    assert res.value == "PEER_ASSOCIATED"
