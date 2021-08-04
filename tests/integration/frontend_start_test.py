# stdlib
import os

# third party
import pytest
import requests

here = os.path.dirname(__file__)

PORT = 9081


@pytest.mark.integration
# def test_serves_frontend(grid_stack: Any) -> None:
def test_serves_frontend() -> None:
    title_str = "<title>PyGrid UI</title>"
    url = f"http://localhost:{PORT}"
    result = requests.get(url)
    assert title_str in result.text
