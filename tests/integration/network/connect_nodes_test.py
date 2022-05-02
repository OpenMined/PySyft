# future
from __future__ import annotations

# stdlib
import os
import subprocess
import time

# third party
import pytest
import requests

# syft absolute
import syft as sy

NETWORK_PORT = 9081
HOST_IP = os.environ.get("HOST_IP", "localhost")
NETWORK_PUBLIC_HOST = f"{HOST_IP}:{NETWORK_PORT}"
print("Network IP", NETWORK_PUBLIC_HOST)
DOMAIN1_PORT = 9082
DOMAIN2_PORT = 9083
NETWORK_VPN_IP = "100.64.0.1"
DOMAIN1_VPN_IP = "100.64.0.2"
DOMAIN2_VPN_IP = "100.64.0.3"
TEST_ROOT_EMAIL = "info@openmined.org"
TEST_ROOT_PASS = "changethis"
CONTAINER_HOST = os.environ.get("CONTAINER_HOST", "docker")
print("CONTAINER_HOST", CONTAINER_HOST)


def join_to_network_python(
    email: str, password: str, port: int, network_host: str
) -> None:
    root_client = sy.login(email=email, password=password, port=port)

    # test Syft API
    root_client.join_network(host_or_ip=network_host)

    # wait for tailscale to connect
    retry_time = 20
    while retry_time > 0:
        retry_time -= 1
        # check network has auto connected
        response = root_client.vpn_status()
        status = response["status"]
        host = response["host"]
        if status == "ok" and "ip" in host:
            break
        time.sleep(1)

    response = root_client.vpn_status()
    return response


def join_to_network_rest(
    email: str, password: str, port: int, network_host: str
) -> None:
    grid_url = sy.grid.GridURL(port=port, path="/api/v1/login")
    if sy.util.ssl_test():
        grid_url = grid_url.to_tls()
    auth_response = requests.post(
        grid_url.url, json={"email": email, "password": password}
    )
    auth = auth_response.json()

    # test HTTP API
    grid_url.path = f"/api/v1/vpn/join/{network_host}"
    headers = {"Authorization": f"Bearer {auth['access_token']}"}
    response = requests.post(grid_url.url, headers=headers)

    result = response.json()
    return result


def run_network_tests(port: int, hostname: str, vpn_ip: str) -> None:
    response = join_to_network_python(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=port,
        network_host=NETWORK_PUBLIC_HOST,
    )

    assert response["status"] == "ok"
    host = response["host"]
    if "ip" not in host:
        print(response)
    assert host["ip"] == vpn_ip
    assert host["hostname"].replace("-", "_") == hostname  # kubernetes forces - not _
    assert host["os"] == "linux"

    response = join_to_network_rest(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=port,
        network_host=NETWORK_PUBLIC_HOST,
    )
    if "status" not in response or response["status"] != "ok":
        print(response)
    assert response["status"] == "ok"


def check_network_is_connected(email: str, password: str, port: int) -> None:
    root_client = sy.login(email=email, password=password, port=port)
    response = root_client.vpn_status()
    return response


def disconnect_network() -> None:
    if CONTAINER_HOST == "docker":
        container = "test_network_1-tailscale-1"

        try:
            cmd = f"docker exec -i {container} tailscale down"
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
        except Exception as e:
            print(f"Exception running: {cmd}. {e}")
    else:
        pod = "tailscale-0"
        container = "container-1"
        context = "k3d-test-network-1"
        namespace = "test-network-1"

        try:
            cmd = (
                f"kubectl exec -it {pod} -c {container}  --context {context} "
                + f"--namespace {namespace} -- tailscale down"
            )
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
        except Exception as e:
            print(f"Exception running: {cmd}. {e}")


@pytest.mark.network
def test_auto_connect_network_to_self() -> None:
    # wait for NETWORK_CHECK_INTERVAL to trigger auto reconnect
    retry_time = 20
    while retry_time > 0:
        retry_time -= 1
        # check network has auto connected
        res = check_network_is_connected(
            email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=NETWORK_PORT
        )
        if res["connected"] is True:
            break
        time.sleep(1)
    assert res["connected"] is True

    # disconnect network
    retry_time = 20
    while retry_time > 0:
        retry_time -= 1

        disconnect_network()

        # check network has auto connected
        res = check_network_is_connected(
            email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=NETWORK_PORT
        )
        if res["connected"] is False:
            break
        time.sleep(1)
    assert res["connected"] is False

    # wait for NETWORK_CHECK_INTERVAL to trigger auto reconnect
    retry_time = 20
    while retry_time > 0:
        retry_time -= 1
        # check network has auto connected
        res = check_network_is_connected(
            email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=NETWORK_PORT
        )
        if res["connected"] is True:
            break
        time.sleep(1)
    assert res["connected"] is True


@pytest.mark.network
def test_connect_domain1_to_network() -> None:
    run_network_tests(
        port=DOMAIN1_PORT, hostname="test_domain_1", vpn_ip=DOMAIN1_VPN_IP
    )


@pytest.mark.network
def test_connect_domain2_to_network() -> None:
    run_network_tests(
        port=DOMAIN2_PORT, hostname="test_domain_2", vpn_ip=DOMAIN2_VPN_IP
    )
