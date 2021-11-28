# stdlib
import os

# future
from __future__ import annotations

# third party
import pytest
import requests

# syft absolute
import syft as sy

NETWORK_HOSTNAME = os.environ.get("TEST_NETWORK_HOSTNAME", "test_network_1")
NETWORK_PUBLIC_PORT = int(os.environ.get("TEST_NETWORK_PUBLIC_PORT", 9081))
NETWORK_PUBLIC_HOST = os.environ.get("TEST_NETWORK_PUBLIC_HOST", "docker-host")
NETWORK_INTERNAL_HOST = os.environ.get("TEST_NETWORK_INTERNAL_HOST", NETWORK_PUBLIC_HOST)
NETWORK_INTERNAL_PORT = os.environ.get("TEST_NETWORK_INTERNAL_PORT", NETWORK_PUBLIC_PORT)
DOMAIN1_PORT = 9082
DOMAIN2_PORT = 9083
NETWORK_VPN_IP = "100.64.0.1"
DOMAIN1_VPN_IP = "100.64.0.2"
DOMAIN2_VPN_IP = "100.64.0.3"
TEST_ROOT_EMAIL = "info@openmined.org"
TEST_ROOT_PASS = "changethis"


def join_to_network_python(
    email: str, password: str, node_api_port: int, node_api_host: str, headscale_addr: str
) -> None:
    root_client = sy.login(email=email, password=password, port=node_api_port, url=f"http://{node_api_host}")

    # test Syft API
    root_client.join_network(host_or_ip=headscale_addr)

    response = root_client.vpn_status()
    return response


def join_to_network_rest(
    email: str, password: str, node_api_port: int, node_api_host: str, headscale_addr: str
) -> None:
    grid_url = sy.grid.GridURL(host_or_ip=node_api_host, port=node_api_host, path="/api/v1/login")
    if sy.util.ssl_test():
        grid_url = grid_url.to_tls()
    auth_response = requests.post(
        grid_url.url, json={"email": email, "password": password}
    )
    auth = auth_response.json()

    # test HTTP API
    grid_url.path = f"/api/v1/vpn/join/{headscale_addr}"
    headers = {"Authorization": f"Bearer {auth['access_token']}"}
    response = requests.post(grid_url.url, headers=headers)

    result = response.json()
    return result


def run_network_tests(node_api_port: int, hostname: str, vpn_ip: str, node_api_host: str, headscale_addr: str) -> None:
    response = join_to_network_python(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        node_api_port=node_api_port,
        node_api_host=node_api_host,
        headscale_addr=headscale_addr,
    )

    assert response["status"] == "ok"
    host = response["host"]
    if "ip" not in host:
        print(response)
    assert host["ip"] == vpn_ip
    assert host["hostname"] == hostname
    assert host["os"] == "linux"

    response = join_to_network_rest(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        node_api_port=node_api_port,
        node_api_host=node_api_host,
        headscale_addr=headscale_addr,
    )
    if "status" not in response or response["status"] != "ok":
        print(response)
    assert response["status"] == "ok"


@pytest.mark.network
@pytest.mark.k8s
def test_connect_network_to_network() -> None:
    run_network_tests(
        node_api_port=NETWORK_PUBLIC_PORT,
        hostname=NETWORK_HOSTNAME,
        vpn_ip=NETWORK_VPN_IP,
        node_api_host=NETWORK_PUBLIC_HOST,
        headscale_addr=f"{NETWORK_INTERNAL_HOST}:{NETWORK_INTERNAL_PORT}",
    )

@pytest.mark.network
@pytest.mark.k8s
def test_connect_domain1_to_network() -> None:
    run_network_tests(
        node_api_port=DOMAIN1_PORT,
        hostname="test_domain_1",
        vpn_ip=DOMAIN1_VPN_IP,
        node_api_host="localhost",
        headscale_addr=f"{NETWORK_PUBLIC_HOST}:{NETWORK_PUBLIC_PORT}",
    )

@pytest.mark.network
def test_connect_domain2_to_network() -> None:
    run_network_tests(
        node_api_port=DOMAIN2_PORT,
        hostname="test_domain_2",
        vpn_ip=DOMAIN2_VPN_IP,
        node_api_host="localhost",
        headscale_addr=f"{NETWORK_PUBLIC_HOST}:{NETWORK_PUBLIC_PORT}"
    )
