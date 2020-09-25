# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for syft.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

# stdlib
from typing import Any
from typing import List

# third party
import _pytest
import pytest


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
