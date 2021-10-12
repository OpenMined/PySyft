# stdlib
import os

# third party
import pytest
import requests

here = os.path.dirname(__file__)


TEST_DOMAIN_IP = str(os.environ.get("TEST_DOMAIN_IP", "localhost")).lower()
TEST_DOMAIN_PORT = int(os.environ.get("TEST_DOMAIN_PORT", 80))


@pytest.mark.integration
def test_serves_domain_frontend() -> None:
    title_str = "<title>domain</title>"
    url = f"http://{TEST_DOMAIN_IP}:{TEST_DOMAIN_PORT}/status"
    result = requests.get(url)
    assert title_str in result.text
