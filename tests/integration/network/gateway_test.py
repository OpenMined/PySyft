# stdlib
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
from syft.client.registry import NetworkRegistry
from syft.client.search import SearchResults
from syft.service.dataset.dataset import Dataset
from syft.service.network.node_peer import NodePeer
from syft.service.request.request import Request
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.user.user_roles import ServiceRole


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
    assert isinstance(result, SyftSuccess)
    assert len(domain_client.peers) == 1
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
