# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for syft.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
# stdlib
import os
import re
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple

# third party
import _pytest
import pytest

# syft absolute
import syft as sy


def pytest_addoption(parser: _pytest.config.argparsing.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: _pytest.config.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "fast: mark test as fast to run")
    config.addinivalue_line("markers", "all: all tests")
    config.addinivalue_line("markers", "asyncio: mark test as asyncio")


def pytest_collection_modifyitems(
    config: _pytest.config.Config, items: List[Any]
) -> None:
    # $ pytest -m fast for the fast tests
    # $ pytest -m slow for the slow tests
    # $ pytest -m all for all the tests
    slow_tests = pytest.mark.slow
    fast_tests = pytest.mark.fast
    all_tests = pytest.mark.all
    for item in items:
        item.add_marker(all_tests)
        if "slow" in item.keywords:
            item.add_marker(slow_tests)
        else:
            item.add_marker(fast_tests)


@pytest.fixture
def duet_wrapper() -> Generator[Tuple[sy.Duet, str], None, None]:
    address = "127.0.0.1"
    allowed_port_range = set(range(5001, 5500))

    class DuetWrapper:
        def __init__(self) -> None:
            self.port = 5001

            """
                If running the tests using multiple processes/threads
                we need to open the duet on different ports such that
                we would not have a port collision
            """

            worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
            worker_number = re.search(r"\d$", worker_name)

            """ The workers have the label master or gw[number] - we extract the number
            and use that as offset for a new port where "number" starts from 0
            """
            offset_port = 0
            if worker_number:
                offset_port = 1 + int(worker_number.group())

            self.port += offset_port

            if self.port not in allowed_port_range:
                raise ValueError(
                    f"The port {self.port} is not in the range [5001:5500)"
                )

            self.url = f"http://{address}:{self.port}/"

        def __enter__(self) -> Tuple[sy.Duet, str]:
            self.duet = sy.Duet(host=address, port=self.port)
            return self.duet, self.url

        def __exit__(self, *args: List[Any]) -> bool:
            self.duet.stop()
            return True

    with DuetWrapper() as duet_wrapper:
        yield duet_wrapper
