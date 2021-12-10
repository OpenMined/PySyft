# stdlib
import os

# third party
import pytest
import requests

from tests.integration.conftest import TestNodeData

here = os.path.dirname(__file__)

@pytest.mark.frontend
def test_serves_domain_frontend(test_domain_1: TestNodeData) -> None:
    title_str = "<title>PyGrid UI</title>"
    url = str(test_domain_1.grid_api_url.with_path("/status"))
    result = requests.get(url)
    assert title_str in result.text


# @pytest.mark.frontend
# def test_serves_network_frontend() -> None:
#     title_str = "<title>network</title>"
#     url = f"http://localhost:{NETWORK_PORT}/status"
#     result = requests.get(url)
#     assert title_str in result.text
