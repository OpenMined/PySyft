# stdlib
import os

# third party
import pytest
import requests

here = os.path.dirname(__file__)

HOST_IP = os.environ.get("HOST_IP", "localhost")


@pytest.mark.frontend
def test_serves_datasite_frontend(datasite_1_port) -> None:
    title_str = "Syft UI"
    url = f"http://{HOST_IP}:{datasite_1_port}"
    result = requests.get(url)
    assert result.status_code == 200
    assert title_str in result.text


@pytest.mark.frontend
def test_serves_network_frontend(gateway_port) -> None:
    title_str = "Syft UI"
    url = f"http://localhost:{gateway_port}"
    result = requests.get(url)
    assert result.status_code == 200
    assert title_str in result.text
