# stdlib
import os

# third party
import pytest
import requests

here = os.path.dirname(__file__)

NETWORK_PORT = 9081
DATASITE_PORT = 9082
HOST_IP = os.environ.get("HOST_IP", "localhost")


@pytest.mark.frontend
def test_serves_datasite_frontend() -> None:
    title_str = "Syft UI"
    url = f"http://{HOST_IP}:{DATASITE_PORT}"
    result = requests.get(url)
    assert result.status_code == 200
    assert title_str in result.text


@pytest.mark.frontend
def test_serves_network_frontend() -> None:
    title_str = "Syft UI"
    url = f"http://localhost:{NETWORK_PORT}"
    result = requests.get(url)
    assert result.status_code == 200
    assert title_str in result.text
