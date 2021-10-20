"""Configuration file to share fixtures across benchmarks."""

# stdlib
from typing import Any
from typing import Callable
from typing import List

# third party
import pytest

# syft absolute
import syft as sy

PORT = 9082
PARTIES = 3
clients = []
for i in range(PARTIES):
    client = sy.login(
        email="info@openmined.org", password="changethis", port=(PORT + i)
    )
    clients.append(client)


@pytest.fixture
def get_clients() -> Callable[[int], List[Any]]:
    def _helper_get_clients(nr_clients: int) -> List[Any]:
        return clients[:nr_clients]

    return _helper_get_clients
