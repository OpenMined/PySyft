"""Configuration file to share fixtures across benchmarks."""

# stdlib
from typing import Any
from typing import Callable
from typing import List

# third party
import _pytest
import pytest

# syft absolute
import syft as sy

clients = []


def login_clients() -> None:
    PORT = 9082
    PARTIES = 3
    for i in range(PARTIES):
        try:
            client = sy.login(
                email="info@openmined.org",
                password="changethis",
                port=(PORT + i),
                verbose=False,
            )
            clients.append(client)
        except Exception as e:
            print(f"Cant connect to client {i}. We might have less running. {e}")


@pytest.fixture
def get_clients() -> Callable[[int], List[Any]]:
    if not clients:
        login_clients()

    def _helper_get_clients(nr_clients: int) -> List[Any]:
        return clients[:nr_clients]

    return _helper_get_clients


def pytest_configure(config: _pytest.config.Config) -> None:
    config.addinivalue_line("markers", "general: general integration tests")
    config.addinivalue_line("markers", "frontend: frontend integration tests")
    config.addinivalue_line("markers", "smpc: smpc integration tests")
    config.addinivalue_line("markers", "network: network integration tests")
    config.addinivalue_line("markers", "k8s: kubernetes integration tests")
    config.addinivalue_line("markers", "e2e: end-to-end integration tests")
    config.addinivalue_line("markers", "security: security integration tests")
