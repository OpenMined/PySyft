# future
from __future__ import annotations

# stdlib
import os
import subprocess
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
import pytest
import requests

# syft absolute
import syft as sy

NETWORK_PORT = 9081
NETWORK_TLS_PORT = 444
HOST_IP = os.environ.get("HOST_IP", "localhost")
NETWORK_PUBLIC_HOST = f"{HOST_IP}:{NETWORK_PORT}"
print("Network IP", NETWORK_PUBLIC_HOST)
DOMAIN1_PORT = 9082
DOMAIN1_TLS_PORT = 445
DOMAIN2_PORT = 9083
DOMAIN2_TLS_PORT = 446
NETWORK_VPN_IP = "100.64.0.1"
DOMAIN1_VPN_IP = "100.64.0.2"
TEST_ROOT_EMAIL = "info@openmined.org"
TEST_ROOT_PASS = "changethis"
CONTAINER_HOST = os.environ.get("CONTAINER_HOST", "docker")
print("CONTAINER_HOST", CONTAINER_HOST)
PROTOCOL = "https" if sy.util.ssl_test() else "http"
print("PROTOCOL", PROTOCOL)


def join_to_network_python(
    email: str, password: str, port: int, network_host: str
) -> Optional[Dict[str, Any]]:
    root_client = sy.login(email=email, password=password, port=port)

    # test Syft API
    try:
        root_client.join_network(host_or_ip=network_host)
    except Exception as e:
        print(e)
        time.sleep(10)
        print("Retrying...")
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
            return response
        time.sleep(3)
    return None


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

    grid_url.path = f"/api/v1/vpn/join/{network_host}"
    headers = {"Authorization": f"Bearer {auth['access_token']}"}
    response = requests.post(grid_url.url, headers=headers)
    result = response.json()
    if "status" in result and result["status"] == "ok":
        return result

    # test HTTP API
    # wait for tailscale to connect
    retry_time = 20
    while retry_time > 0:
        retry_time -= 1
        grid_url.path = "/api/v1/vpn/status"
        headers = {"Authorization": f"Bearer {auth['access_token']}"}
        response = requests.get(grid_url.url, headers=headers)
        result = response.json()
        if "status" in result and result["status"] == "ok":
            return result
        time.sleep(3)
    return None


def run_network_test_python(port: int, hostname: str, vpn_ip: str) -> None:
    retry_time = 3
    while retry_time > 0:
        retry_time -= 1

        try:
            response = join_to_network_python(
                email=TEST_ROOT_EMAIL,
                password=TEST_ROOT_PASS,
                port=port,
                network_host=NETWORK_PUBLIC_HOST,
            )
            if response is not None:
                break
            else:
                time.sleep(10)
        except Exception:
            print(f"failed to run_network_test_python: {retry_time}")
            time.sleep(10)

    assert response["status"] == "ok"
    host = response["host"]
    if "ip" not in host:
        print(response)
    assert host["ip"] == vpn_ip
    assert host["hostname"].replace("-", "_") == hostname.replace(
        "-", "_"
    )  # kubernetes forces - not _
    assert host["os"] == "linux"


def run_network_test_rest(port: int) -> None:
    retry_time = 3
    while retry_time > 0:
        retry_time -= 1

        try:
            response = join_to_network_rest(
                email=TEST_ROOT_EMAIL,
                password=TEST_ROOT_PASS,
                port=port,
                network_host=NETWORK_PUBLIC_HOST,
            )
            if response is not None:
                break
            else:
                time.sleep(10)
        except Exception:
            print(f"failed to run_network_test_rest: {retry_time}")
            time.sleep(10)

    if "status" not in response or response["status"] != "ok":
        print(response)
    assert response["status"] == "ok"


def check_node_is_connected(email: str, password: str, port: int) -> None:
    root_client = sy.login(email=email, password=password, port=port)
    response = root_client.vpn_status()
    return response


@pytest.mark.network
def test_check_settings_fields() -> None:
    domain = sy.login(
        email="info@openmined.org",
        password="changethis",
        port=DOMAIN1_PORT,
    )

    # Assert placeholder and service respone are the same
    assert domain.settings == domain.get_setup().content

    domain_settings_keys = list(domain.settings.keys())

    expected_keys = [
        "domain_name",
        "description",
        "contact",
        "daa",
        "node_uid",
        "tags",
        "deployed_on",
        "use_blob_storage",
    ]

    # Be sure that there's any additional field than the expected ones.
    if sy.__version__ == "0.7.0":
        assert 10 == len(domain_settings_keys)
    else:
        assert len(expected_keys) == len(domain_settings_keys)

    # Be sure that all the expected fields are there.
    for key in expected_keys:
        if sy.__version__ == "0.7.0":
            if key == "node_uid":
                key = "id"
        assert key in domain_settings_keys


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
        pod = "proxy-0"
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


def exec_node_command(command: str, node_name: str) -> None:
    if CONTAINER_HOST == "docker":
        container = node_name + "-tailscale-1"

        try:
            cmd = f"docker exec -i {container} {command}"
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
        except Exception as e:
            print(f"Exception running: {cmd}. {e}")
    else:
        pod = "proxy-0"
        container = "container-1"
        context = "k3d-" + node_name.replace("_", "-")
        namespace = node_name.replace("_", "-")

        try:
            cmd = (
                f"kubectl exec -it {pod} -c {container}  --context {context} "
                + f"--namespace {namespace} -- {command}"
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
        res = check_node_is_connected(
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
        res = check_node_is_connected(
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
        res = check_node_is_connected(
            email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=NETWORK_PORT
        )
        if res["connected"] is True:
            break
        time.sleep(1)
    assert res["connected"] is True


def exchange_credentials(
    email: str, password: str, port: int, network_host: str
) -> None:
    root_client = sy.login(email=email, password=password, port=port)

    # test Syft API
    try:
        response = root_client.networking.initiate_exchange_credentials(
            client=network_host
        )
    except Exception as e:
        print(e)
        time.sleep(10)
        print("Retrying...")
        response = root_client.networking.initiate_exchange_credentials(
            client=network_host
        )

    return response


def add_route(
    email: str,
    password: str,
    port: int,
    network_host: str,
    source_node_url: str,
    private: bool = False,
    autodetect: bool = False,  # 🟡 TODO 20: Change this back to True after building it
) -> None:
    root_client = sy.login(email=email, password=password, port=port)

    # test Syft API
    try:
        response = root_client.networking.add_route_for(
            client=network_host,
            source_node_url=source_node_url,
            private=private,
            autodetect=autodetect,
        )
    except Exception as e:
        print(e)
        time.sleep(10)
        print("Retrying...")
        response = root_client.networking.add_route_for(
            client=network_host, source_node_url=source_node_url, private=private
        )

    return response


def get_routes(
    email: str,
    password: str,
    port: int,
    network_host: str,
    timeout: Optional[int] = None,
) -> List[Dict]:
    root_client = sy.login(email=email, password=password, port=port)

    try:
        response = root_client.networking.list_routes(
            client=network_host, timeout=timeout
        )
    except Exception as e:
        print(e)
        time.sleep(10)
        print("Retrying...")
        response = root_client.networking.list_routes(client=network_host)

    return response.routes_list


@pytest.mark.network
def test_1_exchange_credentials_domain1_to_network() -> None:
    exchange_credentials(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=DOMAIN1_PORT,
        network_host=NETWORK_PUBLIC_HOST,
    )

    routes = get_routes(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=DOMAIN1_PORT,
        network_host=NETWORK_PUBLIC_HOST,
        timeout=60,
    )

    assert isinstance(routes, list)
    assert len(routes) >= 0  # can run this test multiple times


def get_vpn_status(domain: sy.Domain) -> Optional[Dict[str, Any]]:
    status = domain.vpn_status()
    if isinstance(status, dict) and "connected" in status.keys():
        return domain.vpn_status()["connected"]
    return None


@pytest.mark.network
def test_reconnect_domain_node() -> None:
    domain = sy.login(email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=DOMAIN1_PORT)
    network = sy.login(
        email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=NETWORK_PORT
    )

    retry_time = 3
    while retry_time > 0:
        print(f"test_reconnect_domain_node attempt: {retry_time}")
        retry_time -= 1

        try:
            domain.apply_to_network(network)
            status = get_vpn_status(
                domain=domain,
            )
            if status:
                break
            else:
                time.sleep(10)
        except Exception as e:
            print(f"test_reconnect_domain_node failed. {e}")

    # connected
    assert status is True

    retry_time = 3
    while retry_time > 0:
        print(f"test_reconnect_domain_node attempt: {retry_time}")
        retry_time -= 1

        try:
            # Disconnect Domain Node
            exec_node_command(command="tailscale down", node_name="test_domain_1")
            status = get_vpn_status(
                domain=domain,
            )
            if not status:
                break
            else:
                time.sleep(10)
        except Exception as e:
            print(f"test_reconnect_domain_node failed. {e}")

    # Verify if it's really disconnected
    assert not status

    # wait for DOMAIN_CHECK_INTERVAL to trigger auto reconnect
    retry_time = 20
    while retry_time > 0:
        print(f"test_reconnect_domain_node attempt: {retry_time}")
        retry_time -= 1
        # check network has auto connected
        try:
            res = check_node_is_connected(
                email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=DOMAIN1_PORT
            )
            if res["connected"] is True:
                break
            time.sleep(10)
        except Exception as e:
            print(f"test_reconnect_domain_node failed. {e}")
    assert res["connected"] is True


@pytest.mark.network
def test_2_add_route_domain1_to_network() -> None:
    SOURCE_NODE_URL_PORT = DOMAIN1_TLS_PORT if sy.util.ssl_test() else DOMAIN1_PORT
    source_node_url = f"{PROTOCOL}://localhost:{SOURCE_NODE_URL_PORT}"

    add_route(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=DOMAIN1_PORT,
        network_host=NETWORK_PUBLIC_HOST,
        source_node_url=source_node_url,
        private=False,
        autodetect=False,
    )

    routes = get_routes(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=DOMAIN1_PORT,
        network_host=NETWORK_PUBLIC_HOST,
        timeout=60,
    )

    matching_route = None
    for route in routes:
        if route["port"] == SOURCE_NODE_URL_PORT:
            matching_route = route
            break

    assert matching_route["port"] == SOURCE_NODE_URL_PORT
    assert matching_route["is_vpn"] is False
    assert matching_route["private"] is False
    assert matching_route["protocol"] == PROTOCOL
    assert matching_route["host_or_ip"] == "localhost"


@pytest.mark.network
def test_3a_connect_domain1_to_network_vpn() -> None:
    run_network_test_python(
        port=DOMAIN1_PORT, hostname="test_domain_1", vpn_ip=DOMAIN1_VPN_IP
    )


@pytest.mark.network
def test_3b_connect_domain1_to_network_vpn() -> None:
    run_network_test_rest(port=DOMAIN1_PORT)


@pytest.mark.network
def test_4_exchange_credentials_domain2_to_network() -> None:
    exchange_credentials(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=DOMAIN2_PORT,
        network_host=NETWORK_PUBLIC_HOST,
    )

    routes = get_routes(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=DOMAIN2_PORT,
        network_host=NETWORK_PUBLIC_HOST,
    )

    assert isinstance(routes, list)
    assert len(routes) >= 0  # can run this test multiple times


@pytest.mark.network
def test_5_add_route_domain2_to_network() -> None:
    SOURCE_NODE_URL_PORT = DOMAIN2_TLS_PORT if sy.util.ssl_test() else DOMAIN2_PORT
    source_node_url = f"{PROTOCOL}://localhost:{SOURCE_NODE_URL_PORT}"
    add_route(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=DOMAIN2_PORT,
        network_host=NETWORK_PUBLIC_HOST,
        source_node_url=source_node_url,
        private=False,
        autodetect=False,
    )

    routes = get_routes(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=DOMAIN2_PORT,
        network_host=NETWORK_PUBLIC_HOST,
        timeout=60,
    )

    matching_route = None
    for route in routes:
        if route["port"] == SOURCE_NODE_URL_PORT:
            matching_route = route
            break

    assert matching_route["port"] == SOURCE_NODE_URL_PORT
    assert matching_route["is_vpn"] is False
    assert matching_route["private"] is False
    assert matching_route["protocol"] == PROTOCOL
    assert matching_route["host_or_ip"] == "localhost"
