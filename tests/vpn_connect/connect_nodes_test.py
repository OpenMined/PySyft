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


@pytest.mark.integration
def test_connect_network_to_network() -> None:
    root_client = sy.login(
        email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=NETWORK_PORT
    )

    # test Syft API
    root_client.join_network(host_or_ip=NETWORK_PUBLIC_HOST)

    response = root_client.vpn_status()
    assert response["status"] == "ok"
    host = response["host"]
    print(host)
    # assert host["ip"] == NETWORK_VPN_IP
    # assert host["hostname"] == "test_network_1"
    # assert host["os"] == "linux"

#     url = f"http://localhost:{NETWORK_PORT}/api/v1/login"
#     auth_response = requests.post(
#         url, json={"email": TEST_ROOT_EMAIL, "password": TEST_ROOT_PASS}
#     )
#     auth = auth_response.json()

#     # test HTTP API
#     url = f"http://localhost:{NETWORK_PORT}/api/v1/vpn/join/{NETWORK_PUBLIC_HOST}"
#     headers = {"Authorization": f"Bearer {auth['access_token']}"}
#     response = requests.post(url, headers=headers)

#     result = response.json()
#     assert result["status"] == "ok"


# @pytest.mark.integration
# def test_connect_domain1_to_network() -> None:
#     root_client = sy.login(
#         email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=DOMAIN1_PORT
#     )

#     # test Syft API
#     root_client.join_network(host_or_ip=NETWORK_PUBLIC_HOST)

#     response = root_client.vpn_status()
#     assert response["status"] == "ok"
#     host = response["host"]
#     assert host["ip"] == DOMAIN1_VPN_IP
#     assert host["hostname"] == "test_domain_1"
#     assert host["os"] == "linux"

#     url = f"http://localhost:{DOMAIN1_PORT}/api/v1/login"
#     auth_response = requests.post(
#         url, json={"email": TEST_ROOT_EMAIL, "password": TEST_ROOT_PASS}
#     )
#     auth = auth_response.json()

#     # test HTTP API
#     url = f"http://localhost:{DOMAIN1_PORT}/api/v1/vpn/join/{NETWORK_PUBLIC_HOST}"
#     headers = {"Authorization": f"Bearer {auth['access_token']}"}
#     response = requests.post(url, headers=headers)

#     result = response.json()
#     assert result["status"] == "ok"


# @pytest.mark.integration
# def test_connect_domain2_to_network() -> None:
#     root_client = sy.login(
#         email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS, port=DOMAIN2_PORT
#     )

#     # test Syft API
#     root_client.join_network(host_or_ip=NETWORK_PUBLIC_HOST)

#     response = root_client.vpn_status()
#     assert response["status"] == "ok"
#     host = response["host"]
#     assert host["ip"] == DOMAIN2_VPN_IP
#     assert host["hostname"] == "test_domain_2"
#     assert host["os"] == "linux"

#     url = f"http://localhost:{DOMAIN2_PORT}/api/v1/login"
#     auth_response = requests.post(
#         url, json={"email": TEST_ROOT_EMAIL, "password": TEST_ROOT_PASS}
#     )
#     auth = auth_response.json()

#     # test HTTP API
#     url = f"http://localhost:{DOMAIN2_PORT}/api/v1/vpn/join/{NETWORK_PUBLIC_HOST}"
#     headers = {"Authorization": f"Bearer {auth['access_token']}"}
#     response = requests.post(url, headers=headers)

#     result = response.json()
#     assert result["status"] == "ok"
