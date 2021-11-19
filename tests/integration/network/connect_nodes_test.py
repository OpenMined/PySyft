# future
from __future__ import annotations

# stdlib
from urllib.parse import urlparse

# third party
import pytest
import requests

# syft absolute
import syft as sy

NETWORK_PORT = 9081
NETWORK_PUBLIC_HOST = f"docker-host:{NETWORK_PORT}"
DOMAIN1_PORT = 9082
DOMAIN2_PORT = 9083
NETWORK_VPN_IP = "100.64.0.1"
DOMAIN1_VPN_IP = "100.64.0.2"
DOMAIN2_VPN_IP = "100.64.0.3"
TEST_ROOT_EMAIL = "info@openmined.org"
TEST_ROOT_PASS = "changethis"


class GridURL:
    @staticmethod
    def from_url(url: str) -> GridURL:
        try:
            # urlparse doesnt handle no protocol properly
            if "://" not in url:
                url = "http://" + url
            parts = urlparse(url)
            host_or_ip_parts = parts.netloc.split(":")
            # netloc is host:port
            port = 80
            if len(host_or_ip_parts) > 1:
                port = int(host_or_ip_parts[1])
            host_or_ip = host_or_ip_parts[0]
            return GridURL(
                host_or_ip=host_or_ip, path=parts.path, port=port, protocol=parts.scheme
            )
        except Exception as e:
            print(f"Failed to convert url: {url} to GridURL. {e}")

    def __init__(
        self,
        protocol: str = "http",
        host_or_ip: str = "localhost",
        port: int = 80,
        path: str = "",
    ) -> None:
        self.host_or_ip = host_or_ip
        self.path = path
        self.port = port
        self.protocol = protocol

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.path}"

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host_or_ip}:{self.port}"

    def to_tls(self) -> GridURL:
        if self.protocol == "https":
            return self

        # TODO: only ignore ssl in dev mode
        r = requests.get(self.base_url, verify=False)  # ignore ssl cert if its fake
        new_base_url = r.url
        if new_base_url.endswith("/"):
            new_base_url = new_base_url[0:-1]
        return GridURL.from_url(url=f"{new_base_url}{self.path}")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.url}>"

    def __str__(self) -> str:
        return self.url


def join_to_network_python(
    email: str, password: str, port: int, network_host: str
) -> None:
    root_client = sy.login(email=email, password=password, port=port)

    # test Syft API
    root_client.join_network(host_or_ip=network_host)

    response = root_client.vpn_status()
    return response


def join_to_network_rest(
    email: str, password: str, port: int, network_host: str
) -> None:
    grid_url = GridURL(port=port, path="/api/v1/login")
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
    assert host["ip"] == vpn_ip
    assert host["hostname"] == hostname
    assert host["os"] == "linux"

    response = join_to_network_rest(
        email=TEST_ROOT_EMAIL,
        password=TEST_ROOT_PASS,
        port=port,
        network_host=NETWORK_PUBLIC_HOST,
    )
    assert response["status"] == "ok"


@pytest.mark.network
def test_connect_network_to_network() -> None:
    run_network_tests(
        port=NETWORK_PORT, hostname="test_network_1", vpn_ip=NETWORK_VPN_IP
    )


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
