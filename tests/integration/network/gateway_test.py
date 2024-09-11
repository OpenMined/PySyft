# stdlib
import itertools
import os
import time
import uuid

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.abstract_server import ServerType
from syft.client.client import HTTPConnection
from syft.client.client import SyftClient
from syft.client.datasite_client import DatasiteClient
from syft.client.gateway_client import GatewayClient
from syft.client.registry import NetworkRegistry
from syft.service.network.association_request import AssociationRequestChange
from syft.service.network.network_service import ServerPeerAssociationStatus
from syft.service.network.routes import HTTPServerRoute
from syft.service.network.routes import ServerRouteType
from syft.service.network.server_peer import ServerPeer
from syft.service.network.server_peer import ServerPeerConnectionStatus
from syft.service.network.utils import PeerHealthCheckTask
from syft.service.request.request import Request
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.user.user_roles import ServiceRole


@pytest.fixture(scope="function")
def set_env_var(
    gateway_port: int,
    gateway_server: str = "testgateway1",
    host_or_ip: str = "localhost",
    protocol: str = "http",
):
    """Set the environment variable for the network registry JSON string."""
    json_string = f"""
        {{
            "2.0.0": {{
                "gateways": [
                    {{
                        "name": "{gateway_server}",
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
    peers: list[ServerPeer] | SyftError = client.api.services.network.get_all_peers()
    for peer in peers:
        client.api.services.network.delete_peer_by_id(peer.id)
    return SyftSuccess(message="All peers removed.")


@pytest.mark.skip(reason="Will be tested when the network registry URL works.")
def test_network_registry_from_url() -> None:
    assert isinstance(sy.gateways, NetworkRegistry)
    assert len(sy.gateways.all_networks) == len(sy.gateways.online_networks) == 1


@pytest.mark.network
def test_network_registry_env_var(set_env_var) -> None:
    assert isinstance(sy.gateways, NetworkRegistry)
    assert len(sy.gateways.all_networks) == len(sy.gateways.online_networks) == 1
    assert isinstance(sy.gateways[0], GatewayClient)
    assert isinstance(sy.gateways[0].connection, HTTPConnection)


@pytest.mark.network
def test_datasite_connect_to_gateway(
    set_env_var, datasite_1_port: int, gateway_port: int
) -> None:
    # check if we can see the online gateways
    assert isinstance(sy.gateways, NetworkRegistry)
    assert len(sy.gateways.all_networks) == len(sy.gateways.online_networks) == 1

    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Try removing existing peers just to make sure
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # Disable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=False)
    assert isinstance(res, SyftSuccess)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, Request)
    assert isinstance(result.changes[0], AssociationRequestChange)

    assert len(datasite_client.peers) == 1
    assert len(gateway_client.peers) == 0

    gateway_client_root = gateway_client.login(
        email="info@openmined.org", password="changethis"
    )
    res = gateway_client_root.api.services.request.get_all()[-1].approve()

    assert len(gateway_client.peers) == 1

    time.sleep(PeerHealthCheckTask.repeat_time * 2 + 1)

    proxy_datasite_client = gateway_client.peers[0]
    datasite_peer = datasite_client.peers[0]

    assert isinstance(proxy_datasite_client, DatasiteClient)
    assert isinstance(datasite_peer, ServerPeer)

    # Datasite's peer is a gateway and vice-versa
    assert datasite_peer.server_type == ServerType.GATEWAY

    assert gateway_client.name == datasite_peer.name
    assert datasite_client.name == proxy_datasite_client.name

    assert len(gateway_client.datasites) == 1
    assert len(gateway_client.enclaves) == 0

    assert proxy_datasite_client.metadata == datasite_client.metadata
    assert proxy_datasite_client.user_role == ServiceRole.NONE

    datasite_client = datasite_client.login(
        email="info@openmined.org", password="changethis"
    )
    proxy_datasite_client = proxy_datasite_client.login(
        email="info@openmined.org", password="changethis"
    )

    assert proxy_datasite_client.logged_in_user == "info@openmined.org"
    assert proxy_datasite_client.user_role == ServiceRole.ADMIN
    assert proxy_datasite_client.credentials == datasite_client.credentials
    assert (
        proxy_datasite_client.api.endpoints.keys()
        == datasite_client.api.endpoints.keys()
    )

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
@pytest.mark.skip(reason="Disabled since the dataset search functionality was removed")
def test_dataset_search(set_env_var, gateway_port: int, datasite_1_port: int) -> None:
    """
    Scenario: Connecting a datasite server to a gateway server. The datasite
        client then upload a dataset, which should be searchable by the syft network.
        People who install syft can see the mock data and metadata of the uploaded datasets
    """
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Try removing existing peers just to make sure
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connect the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    # the datasite client uploads a dataset
    input_data = np.array([1, 2, 3])
    mock_data = np.array([4, 5, 6])
    asset_name = _random_hash()
    asset = sy.Asset(name=asset_name, data=input_data, mock=mock_data)
    dataset_name = _random_hash()
    dataset = sy.Dataset(name=dataset_name, asset_list=[asset])
    dataset_res = datasite_client.upload_dataset(dataset)
    assert isinstance(dataset_res, SyftSuccess)

    # since dataset search is done by checking from the online datasites,
    # we need to wait to make sure peers health check is done
    # time.sleep(PeerHealthCheckTask.repeat_time * 2 + 1)
    # test if the dataset can be searched by the syft network
    # right_search = sy.search(dataset_name)
    # assert isinstance(right_search, SearchResults)
    # assert len(right_search) == 1
    # dataset = right_search[0]
    # assert isinstance(dataset, Dataset)
    # assert len(dataset.assets) == 1
    # assert isinstance(dataset.assets[0].mock, np.ndarray)
    # assert dataset.assets[0].data is None

    # # search a wrong dataset should return an empty list
    # wrong_search = sy.search(_random_hash())
    # assert len(wrong_search) == 0

    # # the datasite client delete the dataset
    # datasite_client.api.services.dataset.delete(uid=dataset.id)

    # # Remove existing peers
    # assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    # assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.skip(reason="Possible bug")
@pytest.mark.network
def test_datasite_gateway_user_code(
    set_env_var, datasite_1_port: int, gateway_port: int
) -> None:
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Try removing existing peers just to make sure
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # the datasite client uploads a dataset
    input_data = np.array([1, 2, 3])
    mock_data = np.array([4, 5, 6])
    asset_name = _random_hash()
    asset = sy.Asset(name=asset_name, data=input_data, mock=mock_data)
    dataset_name = _random_hash()
    dataset = sy.Dataset(name=dataset_name, asset_list=[asset])
    dataset_res = datasite_client.upload_dataset(dataset)
    assert isinstance(dataset_res, SyftSuccess)

    # the datasite client registers a data data scientist account on its datasite
    random_name: str = str(_random_hash())
    user_create_res = datasite_client.register(
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

    # the datasite client connects to the gateway
    gateway_con_res = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(gateway_con_res, SyftSuccess)

    # get the proxy client to the datasite, login to the data scientist account
    proxy_client = gateway_client.datasites[0]
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

    request_res = proxy_ds.code.request_code_execution(mock_function)
    assert isinstance(request_res, Request)

    # datasite client approves the request
    assert len(datasite_client.requests.get_all()) == 1
    req_approve_res = datasite_client.requests[-1].approve()
    assert isinstance(req_approve_res, SyftSuccess)

    # the proxy data scientist client executes the code and gets the result
    result = proxy_ds.code.mock_function(asset=asset)
    final_result = result.get()
    assert (final_result == input_data + 1).all()

    # the datasite client delete the dataset
    datasite_client.api.services.dataset.delete(uid=dataset.id)

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
def test_deleting_peers(set_env_var, datasite_1_port: int, gateway_port: int) -> None:
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # clean up before test
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(datasite_client.peers) == 1
    assert len(gateway_client.peers) == 1

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)
    # check that removing peers work as expected
    assert len(datasite_client.peers) == 0
    assert len(gateway_client.peers) == 0

    # reconnect the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(datasite_client.peers) == 1
    assert len(gateway_client.peers) == 1

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)
    # check that removing peers work as expected
    assert len(datasite_client.peers) == 0
    assert len(gateway_client.peers) == 0


@pytest.mark.network
def test_add_route(set_env_var, gateway_port: int, datasite_1_port: int) -> None:
    """
    Test the network service's `add_route` functionalities to add routes directly
    for a self datasite.
    Scenario: Connect a datasite to a gateway. The gateway adds 2 new routes to the datasite
    and check their priorities get updated.
    Check for the gateway if the proxy client to connect to the datasite uses the
    route with the highest priority.
    """
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Try removing existing peers just to make sure
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(datasite_client.peers) == 1
    assert len(gateway_client.peers) == 1

    # add a new route to connect to the datasite
    new_route = HTTPServerRoute(host_or_ip="localhost", port=10000)
    datasite_peer: ServerPeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=datasite_peer.verify_key, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(datasite_peer.server_routes) == 2
    assert datasite_peer.server_routes[-1].port == new_route.port

    # adding another route to the datasite
    new_route2 = HTTPServerRoute(host_or_ip="localhost", port=10001)
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=datasite_peer.verify_key, route=new_route2
    )
    assert isinstance(res, SyftSuccess)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(datasite_peer.server_routes) == 3
    assert datasite_peer.server_routes[-1].port == new_route2.port
    assert datasite_peer.server_routes[-1].priority == 3

    # add an existed route to the datasite. Its priority should not be updated
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=datasite_peer.verify_key, route=datasite_peer.server_routes[0]
    )
    assert "route already exists" in res.message
    assert isinstance(res, SyftSuccess)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(datasite_peer.server_routes) == 3
    assert datasite_peer.server_routes[0].priority == 1

    # getting the proxy client using the current highest priority route should
    # be successful since now we pick the oldest route (port 9082 with priority 1)
    # to have the highest priority by default
    proxy_datasite_client = gateway_client.peers[0]
    assert isinstance(proxy_datasite_client, DatasiteClient)

    # the routes the datasite client uses to connect to the gateway should stay the same
    gateway_peer: ServerPeer = datasite_client.peers[0]
    assert len(gateway_peer.server_routes) == 1

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
def test_delete_route(set_env_var, gateway_port: int, datasite_1_port: int) -> None:
    """
    Scenario:
    Connect a datasite to a gateway. The gateway adds a new route to the datasite
    and then deletes it.
    """
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Try removing existing peers just to make sure
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(datasite_client.peers) == 1
    assert len(gateway_client.peers) == 1

    # add a new route to connect to the datasite
    new_route = HTTPServerRoute(host_or_ip="localhost", port=10000)
    datasite_peer: ServerPeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=datasite_peer.verify_key, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(datasite_peer.server_routes) == 2
    assert datasite_peer.server_routes[-1].port == new_route.port

    # delete the added route
    res = gateway_client.api.services.network.delete_route(
        peer_verify_key=datasite_peer.verify_key, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert len(datasite_peer.server_routes) == 1
    assert datasite_peer.server_routes[-1].port == datasite_1_port

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
def test_add_route_on_peer(
    set_env_var, gateway_port: int, datasite_1_port: int
) -> None:
    """
    Test the `add_route_on_peer` of network service.
    Connect a datasite to a gateway.
    The gateway adds 2 new routes for itself remotely on the datasite and check their priorities.
    Then the datasite adds a route to itself for the gateway.
    """
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Remove existing peers
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)
    assert len(datasite_client.peers) == 1
    assert len(gateway_client.peers) == 1
    gateway_peer: ServerPeer = datasite_client.peers[0]
    assert len(gateway_peer.server_routes) == 1
    assert gateway_peer.server_routes[-1].priority == 1

    # adding a new route for the datasite
    new_route = HTTPServerRoute(host_or_ip="localhost", port=10000)
    datasite_peer: ServerPeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=datasite_peer, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    gateway_peer = datasite_client.api.services.network.get_all_peers()[0]
    assert len(gateway_peer.server_routes) == 2
    assert gateway_peer.server_routes[-1].port == new_route.port
    assert gateway_peer.server_routes[-1].priority == 2

    # adding another route for the datasite
    new_route2 = HTTPServerRoute(host_or_ip="localhost", port=10001)
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=datasite_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)
    gateway_peer = datasite_client.api.services.network.get_all_peers()[0]
    assert len(gateway_peer.server_routes) == 3
    assert gateway_peer.server_routes[-1].port == new_route2.port
    assert gateway_peer.server_routes[-1].priority == 3

    # the datasite calls `add_route_on_peer` to to add a route to itself for the gateway
    assert len(datasite_peer.server_routes) == 1
    res = datasite_client.api.services.network.add_route_on_peer(
        peer=datasite_client.peers[0], route=new_route
    )
    assert isinstance(res, SyftSuccess)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert datasite_peer.server_routes[-1].port == new_route.port
    assert len(datasite_peer.server_routes) == 2

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_delete_route_on_peer(
    set_env_var, gateway_port: int, datasite_1_port: int
) -> None:
    """
    Connect a datasite to a gateway, the gateway adds 2 new routes for the datasite
    , then delete them.
    """
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Remove existing peers
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    # gateway adds 2 new routes for the datasite
    new_route = HTTPServerRoute(host_or_ip="localhost", port=10000)
    new_route2 = HTTPServerRoute(host_or_ip="localhost", port=10001)
    datasite_peer: ServerPeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=datasite_peer, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=datasite_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)

    gateway_peer: ServerPeer = datasite_client.peers[0]
    assert len(gateway_peer.server_routes) == 3

    # gateway delete the routes for the datasite
    res = gateway_client.api.services.network.delete_route_on_peer(
        peer=datasite_peer, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    gateway_peer = datasite_client.peers[0]
    assert len(gateway_peer.server_routes) == 2

    res = gateway_client.api.services.network.delete_route_on_peer(
        peer=datasite_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)
    gateway_peer = datasite_client.peers[0]
    assert len(gateway_peer.server_routes) == 1

    # gateway deletes the last the route to it for the datasite
    last_route: ServerRouteType = gateway_peer.server_routes[0]
    res = gateway_client.api.services.network.delete_route_on_peer(
        peer=datasite_peer, route=last_route
    )
    assert isinstance(res, SyftSuccess)
    assert "There is no routes left" in res.message
    assert (
        len(datasite_client.peers) == 0
    )  # gateway is no longer a peer of the datasite

    # The gateway client also removes the datasite as a peer
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
def test_update_route_priority(
    set_env_var, gateway_port: int, datasite_1_port: int
) -> None:
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Try remove existing peers
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    # gateway adds 2 new routes to the datasite
    new_route = HTTPServerRoute(host_or_ip="localhost", port=10000)
    new_route2 = HTTPServerRoute(host_or_ip="localhost", port=10001)
    datasite_peer: ServerPeer = gateway_client.api.services.network.get_all_peers()[0]
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=datasite_peer.verify_key, route=new_route
    )
    assert isinstance(res, SyftSuccess)
    res = gateway_client.api.services.network.add_route(
        peer_verify_key=datasite_peer.verify_key, route=new_route2
    )
    assert isinstance(res, SyftSuccess)

    # check if the priorities of the routes are correct
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in datasite_peer.server_routes
    }
    assert routes_port_priority[datasite_1_port] == 1
    assert routes_port_priority[new_route.port] == 2
    assert routes_port_priority[new_route2.port] == 3

    # update the priorities for the routes
    res = gateway_client.api.services.network.update_route_priority(
        peer_verify_key=datasite_peer.verify_key, route=new_route, priority=5
    )
    assert isinstance(res, SyftSuccess)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in datasite_peer.server_routes
    }
    assert routes_port_priority[new_route.port] == 5

    # if we don't specify `priority`, the route will be automatically updated
    # to have the biggest priority value among all routes
    res = gateway_client.api.services.network.update_route_priority(
        peer_verify_key=datasite_peer.verify_key, route=new_route2
    )
    assert isinstance(res, SyftSuccess)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in datasite_peer.server_routes
    }
    assert routes_port_priority[new_route2.port] == 6

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
def test_update_route_priority_on_peer(
    set_env_var, gateway_port: int, datasite_1_port: int
) -> None:
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Remove existing peers
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # Enable automatic acceptance of association requests
    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    # gateway adds 2 new routes to itself remotely on the datasite server
    datasite_peer: ServerPeer = gateway_client.api.services.network.get_all_peers()[0]
    new_route = HTTPServerRoute(host_or_ip="localhost", port=10000)
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=datasite_peer, route=new_route
    )
    assert isinstance(res, SyftSuccess)

    new_route2 = HTTPServerRoute(host_or_ip="localhost", port=10001)
    res = gateway_client.api.services.network.add_route_on_peer(
        peer=datasite_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)

    # check if the priorities of the routes are correct
    gateway_peer = datasite_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in gateway_peer.server_routes
    }
    assert routes_port_priority[gateway_port] == 1
    assert routes_port_priority[new_route.port] == 2
    assert routes_port_priority[new_route2.port] == 3

    # gateway updates the route priorities for the datasite remotely
    res = gateway_client.api.services.network.update_route_priority_on_peer(
        peer=datasite_peer, route=new_route, priority=5
    )
    assert isinstance(res, SyftSuccess)
    res = gateway_client.api.services.network.update_route_priority_on_peer(
        peer=datasite_peer, route=new_route2
    )
    assert isinstance(res, SyftSuccess)

    gateway_peer = datasite_client.api.services.network.get_all_peers()[0]
    routes_port_priority: dict = {
        route.port: route.priority for route in gateway_peer.server_routes
    }
    assert routes_port_priority[new_route.port] == 5
    assert routes_port_priority[new_route2.port] == 6

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
def test_dataset_stream(set_env_var, gateway_port: int, datasite_1_port: int) -> None:
    """
    Scenario: Connecting a datasite server to a gateway server. The datasite
        client then upload a dataset, which should be searchable by the syft network.
        People who install syft can see the mock data and metadata of the uploaded datasets
    """
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Remove existing peers just to make sure
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    res = gateway_client.settings.allow_association_request_auto_approval(enable=True)
    assert isinstance(res, SyftSuccess)

    # connect the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, SyftSuccess)

    # the datasite client uploads a dataset
    input_data = np.array([1, 2, 3])
    mock_data = np.array([4, 5, 6])
    asset_name = _random_hash()
    asset = sy.Asset(name=asset_name, data=input_data, mock=mock_data)
    dataset_name = _random_hash()
    dataset = sy.Dataset(name=dataset_name, asset_list=[asset])
    dataset_res = datasite_client.upload_dataset(dataset)
    assert isinstance(dataset_res, SyftSuccess)

    datasite_proxy_client = next(
        gateway_client.datasites[i]
        for i in itertools.count()
        if gateway_client.datasites[i].name == datasite_client.name
    )
    root_proxy_client = datasite_proxy_client.login(
        email="info@openmined.org", password="changethis"
    )
    retrieved_dataset = root_proxy_client.datasets[dataset_name]
    retrieved_asset = retrieved_dataset.assets[asset_name]
    assert np.all(retrieved_asset.data == input_data)

    # the datasite client delete the dataset
    datasite_client.api.services.dataset.delete(uid=retrieved_dataset.id)

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


# TODO: remove this and fix this test
@pytest.mark.xfail(reason="Unsure but its flapping in CI we need to fix it")
@pytest.mark.network
def test_peer_health_check(
    set_env_var, gateway_port: int, datasite_1_port: int
) -> None:
    """
    Scenario: Connecting a datasite server to a gateway server.
    The gateway client approves the association request.
    The gateway client checks that the datasite peer is associated
    """
    # login to the datasite and gateway
    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    res = gateway_client.settings.allow_association_request_auto_approval(enable=False)
    assert isinstance(res, SyftSuccess)

    # Try removing existing peers just to make sure
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # gateway checks that the datasite is not yet associated
    res = gateway_client.api.services.network.check_peer_association(
        peer_id=datasite_client.id
    )
    assert isinstance(res, ServerPeerAssociationStatus)
    assert res.value == "PEER_NOT_FOUND"

    # the datasite tries to connect to the gateway
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, Request)
    assert isinstance(result.changes[0], AssociationRequestChange)

    # check that the peer's association request is pending
    res = gateway_client.api.services.network.check_peer_association(
        peer_id=datasite_client.id
    )
    assert isinstance(res, ServerPeerAssociationStatus)
    assert res.value == "PEER_ASSOCIATION_PENDING"

    # the datasite tries to connect to the gateway (again)
    result = datasite_client.connect_to_gateway(gateway_client)
    assert isinstance(result, Request)  # the pending request is returned
    # there should be only 1 association requests from the datasite
    assert len(gateway_client.api.services.request.get_all()) == 1

    # check again that the peer's association request is still pending
    res = gateway_client.api.services.network.check_peer_association(
        peer_id=datasite_client.id
    )
    assert isinstance(res, ServerPeerAssociationStatus)
    assert res.value == "PEER_ASSOCIATION_PENDING"

    # the gateway client approves one of the association requests
    res = gateway_client.api.services.request.get_all()[-1].approve()
    assert len(gateway_client.peers) == 1

    # the gateway client checks that the peer is associated
    res = gateway_client.api.services.network.check_peer_association(
        peer_id=datasite_client.id
    )
    assert isinstance(res, ServerPeerAssociationStatus)
    assert res.value == "PEER_ASSOCIATED"

    time.sleep(PeerHealthCheckTask.repeat_time * 2 + 1)
    datasite_peer = gateway_client.api.services.network.get_all_peers()[0]
    assert datasite_peer.ping_status == ServerPeerConnectionStatus.ACTIVE

    # Remove existing peers
    assert isinstance(_remove_existing_peers(datasite_client), SyftSuccess)
    assert isinstance(_remove_existing_peers(gateway_client), SyftSuccess)


@pytest.mark.network
def test_reverse_tunnel_connection(datasite_1_port: int, gateway_port: int):
    # login to the datasite and gateway

    gateway_client: GatewayClient = sy.login(
        port=gateway_port, email="info@openmined.org", password="changethis"
    )
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    _ = gateway_client.settings.allow_association_request_auto_approval(enable=False)

    # Try removing existing peers just to make sure
    _remove_existing_peers(datasite_client)
    _remove_existing_peers(gateway_client)

    # connecting the datasite to the gateway
    result = datasite_client.connect_to_gateway(gateway_client, reverse_tunnel=True)

    assert isinstance(result, Request)
    assert isinstance(result.changes[0], AssociationRequestChange)

    assert len(datasite_client.peers) == 1

    # Datasite's peer is a gateway and vice-versa
    datasite_peer = datasite_client.peers[0]
    assert datasite_peer.server_type == ServerType.GATEWAY
    assert datasite_peer.server_routes[0].rtunnel_token is None
    assert len(gateway_client.peers) == 0

    gateway_client_root = gateway_client.login(
        email="info@openmined.org", password="changethis"
    )
    _ = gateway_client_root.api.services.request.get_all()[-1].approve()
    time.sleep(90)

    gateway_peers = gateway_client.api.services.network.get_all_peers()
    assert len(gateway_peers) == 1
    assert len(gateway_peers[0].server_routes) == 1
    assert gateway_peers[0].server_routes[0].rtunnel_token is not None

    proxy_datasite_client = gateway_client.peers[0]

    assert isinstance(proxy_datasite_client, DatasiteClient)
    assert isinstance(datasite_peer, ServerPeer)
    assert gateway_client.name == datasite_peer.name
    assert datasite_client.name == proxy_datasite_client.name

    assert not isinstance(proxy_datasite_client.datasets.get_all(), SyftError)

    # Try removing existing peers just to make sure
    _remove_existing_peers(gateway_client)
    _remove_existing_peers(datasite_client)
