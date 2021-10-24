# stdlib
from typing import Any
from typing import Dict
import uuid

# third party
import pytest
import requests

# syft absolute
import syft as sy

NETWORK_PORT = 9081
DOMAIN1_PORT = 9082
DOMAIN2_PORT = 9083


def get_user_details(unique_email: str) -> Dict[str, Any]:
    return {
        "name": "Sheldon Cooper",
        "email": unique_email,
        "password": "bazinga",
    }


@pytest.mark.integration
def test_domain1_ping_network() -> None:
    unique_email = f"{uuid.uuid4()}@caltech.edu"

    root_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )
    unique_user = get_user_details(unique_email=unique_email)
    root_client.users.create(**unique_user)

    url = f"http://localhost:{DOMAIN1_PORT}/api/v1/login"
    auth_response = requests.post(
        url, json={"email": unique_user["email"], "password": unique_user["password"]}
    )
    auth = auth_response.json()

    NETWORK_VPN_IP = "100.64.0.1"
    url = f"http://localhost:{DOMAIN1_PORT}/api/v1/ping/{NETWORK_VPN_IP}"
    headers = {"Authorization": f"Bearer {auth['access_token']}"}
    ping_response = requests.get(url, headers=headers)

    result = ping_response.json()
    assert result["kwargs"]["host_or_ip"]["data"].endswith(NETWORK_VPN_IP)
    assert result["kwargs"]["status_code"] == 200


@pytest.mark.integration
def test_domain2_ping_network() -> None:
    unique_email = f"{uuid.uuid4()}@caltech.edu"

    root_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN2_PORT
    )
    unique_user = get_user_details(unique_email=unique_email)
    root_client.users.create(**unique_user)

    url = f"http://localhost:{DOMAIN2_PORT}/api/v1/login"
    auth_response = requests.post(
        url, json={"email": unique_user["email"], "password": unique_user["password"]}
    )
    auth = auth_response.json()

    NETWORK_VPN_IP = "100.64.0.1"
    url = f"http://localhost:{DOMAIN2_PORT}/api/v1/ping/{NETWORK_VPN_IP}"
    headers = {"Authorization": f"Bearer {auth['access_token']}"}
    ping_response = requests.get(url, headers=headers)

    result = ping_response.json()
    assert result["kwargs"]["host_or_ip"]["data"].endswith(NETWORK_VPN_IP)
    assert result["kwargs"]["status_code"] == 200


@pytest.mark.integration
def test_domain1_ping_domain2() -> None:
    unique_email = f"{uuid.uuid4()}@caltech.edu"

    root_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )
    unique_user = get_user_details(unique_email=unique_email)
    root_client.users.create(**unique_user)

    url = f"http://localhost:{DOMAIN1_PORT}/api/v1/login"
    auth_response = requests.post(
        url, json={"email": unique_user["email"], "password": unique_user["password"]}
    )
    auth = auth_response.json()

    DOMAIN2_VPN_IP = "100.64.0.3"
    url = f"http://localhost:{DOMAIN1_PORT}/api/v1/ping/{DOMAIN2_VPN_IP}"
    headers = {"Authorization": f"Bearer {auth['access_token']}"}
    ping_response = requests.get(url, headers=headers)

    result = ping_response.json()
    assert result["kwargs"]["host_or_ip"]["data"].endswith(DOMAIN2_VPN_IP)
    assert result["kwargs"]["status_code"] == 200
