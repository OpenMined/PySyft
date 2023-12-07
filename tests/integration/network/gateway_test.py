# stdlib
from textwrap import dedent
import uuid

# third party
import numpy as np

# syft absolute
import syft as sy
from syft.abstract_node import NodeType
from syft.client.domain_client import DomainClient
from syft.client.gateway_client import GatewayClient
from syft.service.network.node_peer import NodePeer
from syft.service.request.request import Request
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


def random_hash() -> str:
    return uuid.uuid4().hex[:16]


def test_domain_gateway_user_code(domain_1_port, gateway_port):
    gateway_client: GatewayClient = sy.login_as_guest(port=gateway_port)

    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    input_data = np.array([1, 2, 3])
    mock_data = np.array([4, 5, 6])

    asset_name = random_hash()
    asset = sy.Asset(name=asset_name, data=input_data, mock=mock_data)
    dataset_name = random_hash()
    dataset = sy.Dataset(name=dataset_name, asset_list=[asset])

    dataset_res = domain_client.upload_dataset(dataset)

    assert isinstance(dataset_res, SyftSuccess)

    user_create_res = domain_client.register(
        name="Sheldon Cooper",
        email="sheldon@caltech.edu",
        password="changethis",
        password_verify="changethis",
        institution="Caltech",
        website="https://www.caltech.edu/",
    )

    assert isinstance(user_create_res, SyftSuccess)

    gateway_con_res = domain_client.connect_to_gateway(gateway_client)
    assert isinstance(gateway_con_res, SyftSuccess)

    proxy_client = gateway_client.domains[0]

    proxy_ds = proxy_client.login(
        email="sheldon@caltech.edu", password="changethis", password_verify="changethis"
    )

    asset = proxy_ds.datasets[0].assets[0]

    @sy.syft_function_single_use(asset=asset)
    def test_function(asset):
        return asset + 1

    test_function.code = dedent(test_function.code)

    request_res = proxy_ds.code.request_code_execution(test_function)
    assert isinstance(request_res, Request)

    assert len(domain_client.requests.get_all()) == 1

    req_approve_res = domain_client.requests[-1].approve()
    assert isinstance(req_approve_res, SyftSuccess)

    result = proxy_ds.code.test_function(asset=asset)

    final_result = result.get()

    assert (final_result == input_data + 1).all()
