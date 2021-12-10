# future
from __future__ import annotations

# stdlib
import os

# third party
import pytest
import requests

# syft absolute
import syft as sy
from tests.integration.conftest import TestNodeData

TEST_ROOT_EMAIL = "info@openmined.org"
TEST_ROOT_PASS = "changethis"


def join_to_network_python(
    email: str, password: str, node: TestNodeData,
) -> None:
    root_client = sy.login(email=email, password=password, url=node.grid_api_url)

    # test Syft API
    root_client.join_network(host_or_ip=node.headscale_addr)

    response = root_client.vpn_status()
    return response


def join_to_network_rest(
    email: str, password: str, node: TestNodeData
) -> None:
    grid_url = node.grid_api_url.with_path("/api/v1/login")
    if sy.util.ssl_test():
        grid_url = grid_url.to_tls()
    auth_response = requests.post(
        grid_url.url, json={"email": email, "password": password}
    )
    auth = auth_response.json()

    # test HTTP API
    grid_url.path = f"/api/v1/vpn/join/{node.headscale_addr}"
    headers = {"Authorization": f"Bearer {auth['access_token']}"}
    response = requests.post(grid_url.url, headers=headers)

    result = response.json()
    return result


def run_network_tests(node: TestNodeData) -> None:
    response = join_to_network_python(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        node=node
    )

    assert response["status"] == "ok"
    host = response["host"]
    if "ip" not in host:
        print(response)
    assert host["ip"] == node.vpn_ip
    assert host["hostname"] == node.hostname
    assert host["os"] == "linux"

    response = join_to_network_rest(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        node=node
    )
    if "status" not in response or response["status"] != "ok":
        print(response)
    assert response["status"] == "ok"


@pytest.mark.network
def test_connect_network_to_network(test_network: TestNodeData) -> None:
    run_network_tests(test_network)

@pytest.mark.network
def test_connect_domain1_to_network(test_domain_1: TestNodeData) -> None:
    run_network_tests(test_domain_1)

@pytest.mark.network
def test_connect_domain2_to_network(test_domain_2: TestNodeData) -> None:
    run_network_tests(test_domain_2)
