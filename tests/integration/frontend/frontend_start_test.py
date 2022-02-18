# stdlib
import os

# third party
import pytest
import requests

here = os.path.dirname(__file__)

NETWORK_PORT = 9081
DOMAIN_PORT = 9082


@pytest.mark.frontend
def test_serves_domain_frontend() -> None:
    title_str = "<title>PyGrid UI</title>"
    url = f"http://localhost:{DOMAIN_PORT}/status"
    result = requests.get(url)
    assert title_str in result.text


# @pytest.mark.frontend
# def test_serves_network_frontend() -> None:
#     title_str = "<title>network</title>"
#     url = f"http://localhost:{NETWORK_PORT}/status"
#     result = requests.get(url)
#     assert title_str in result.text
