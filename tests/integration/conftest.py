"""Configuration file to share fixtures across benchmarks."""

# stdlib
from typing import Any
from typing import Callable
from typing import List
import os
import subprocess

from requests.api import head

# third party
import _pytest
import pytest

# syft absolute
import syft as sy
from syft.grid import GridURL

clients = []

class TestNodeData:
    __test__ = False

    def __init__(self, grid_api_url: GridURL, headscale_addr: str, network_hostname: str, vpn_ip: str):
        self.grid_api_url = grid_api_url
        self.headscale_addr = headscale_addr
        self.hostname = network_hostname
        self.vpn_ip = vpn_ip

def login_clients(login_urls: List[GridURL]) -> None:
    for url in login_urls:
        try:
            client = sy.login(
                email="info@openmined.org",
                password="changethis",
                url=url,
                verbose=False,
            )
            clients.append(client)
        except Exception as e:
            print(f"Cant connect to client {url}. We might have less running. {e}")


@pytest.fixture
def get_clients(test_network:TestNodeData, test_domain_1: TestNodeData, test_domain_2: TestNodeData) -> Callable[[int], List[Any]]:
    if not clients:
        urls = [test_network.grid_api_url, test_domain_1.grid_api_url, test_domain_2.grid_api_url]
        login_clients(urls)

    def _helper_get_clients(nr_clients: int) -> List[Any]:
        return clients[:nr_clients]

    return _helper_get_clients

@pytest.fixture(scope="session")
def test_network() -> TestNodeData:
    network_deployed_in = os.environ.get("TEST_NETWORK_DEPLOYED_IN", "docker-compose")
    vpn_ip = "100.64.0.1"
    if network_deployed_in == 'minikube':
        name = "syft-network"
        public_host = minikube_ip(name)
        public_port = k_node_port(name)
        grid_api_url = GridURL(host_or_ip=public_host, port=public_port)

        # To connect the network node to its internal headscale,
        # tailscale should be given the host on the cluster internal network
        network_internal_ip = k_cluster_ip(name)
        network_internal_port = k_service_port(name)
        network_internal_host = f"{network_internal_ip}:{network_internal_port}"

        network_hostname = "tailscale-0"

        return TestNodeData(grid_api_url, network_internal_host, network_hostname, vpn_ip)

    docker_grid_api_url = GridURL(host_or_ip="docker-host", port=9081)
    docker_headscale_addr = "test_network_1:9081"
    docker_network_hostname = "test_network_1"
    return TestNodeData(docker_grid_api_url, docker_headscale_addr, docker_network_hostname, vpn_ip)

@pytest.fixture(scope="session")
def test_domain_1() -> TestNodeData:
    domain_deployed_in = os.environ.get("TEST_DOMAIN_DEPLOYED_IN", "docker-compose")
    vpn_ip = "100.64.0.2"
    if domain_deployed_in == 'minikube':
        name = "syft-domain"
        public_host = minikube_ip(name)
        public_port = k_node_port(name)
        return TestNodeData(
            grid_api_url=GridURL(host_or_ip=public_host, port=public_port),
            headscale_addr="hi",
            # headscale_addr=f"{test_network.grid_api_url.host_or_ip}:{test_network.grid_api_url.port}",
            network_hostname="tailscale-0",
            vpn_ip=vpn_ip)
    return TestNodeData(
        grid_api_url=GridURL(host_or_ip="docker-host", port=9082),
        headscale_addr="test_network_1:9081",
        network_hostname="test_domain_1",
        vpn_ip=vpn_ip)

# Note: test_domain_2 is only deployed in docker-compose so that we can test a setup
# with a network on k8s connected to this domain node running in docker
@pytest.fixture(scope="session")
def test_domain_2(test_network: TestNodeData) -> TestNodeData:
    vpn_ip = "100.64.0.3"
    return TestNodeData(
        grid_api_url=GridURL(host_or_ip="docker-host", port=9083),
        headscale_addr="test_network_1:9081",
        network_hostname="test_domain_2",
        vpn_ip=vpn_ip)

def minikube_ip(profile: str) -> str:
    return subprocess.check_output(['minikube', 'ip', '--profile', profile], text=True).strip()

def k_cluster_ip(context: str) -> str:
    return subprocess.check_output([
        'kubectl',
        '--context', context,
        '--namespace', 'openmined',
        'get', 'svc', 'traefik',
        '-o', 'jsonpath=\'{.spec.clusterIP}\'',
    ], text=True).strip().strip("'")

def k_node_port(context: str) -> str:
    return subprocess.check_output([
        'kubectl',
        '--context', context,
        '--namespace', 'openmined',
        'get', 'svc', 'traefik',
        '-o', 'jsonpath=\'{.spec.ports[0].nodePort}\'',
    ], text=True).strip().strip("'")

def k_service_port(context: str) -> str:
    return subprocess.check_output([
        'kubectl',
        '--context', context,
        '--namespace', 'openmined',
        'get', 'svc', 'traefik',
        '-o' 'jsonpath=\'{.spec.ports[0].port}\'',
    ], text=True).strip().strip("'")

def pytest_configure(config: _pytest.config.Config) -> None:
    config.addinivalue_line("markers", "general: general integration tests")
    config.addinivalue_line("markers", "frontend: frontend integration tests")
    config.addinivalue_line("markers", "smpc: smpc integration tests")
    config.addinivalue_line("markers", "network: network integration tests")
    config.addinivalue_line("markers", "domain: domain integration tests")
    config.addinivalue_line("markers", "k8s: kubernetes integration tests")
    config.addinivalue_line("markers", "e2e: end-to-end integration tests")
    config.addinivalue_line("markers", "security: security integration tests")
