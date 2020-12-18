# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for syft.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

# stdlib
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import List as TypeList

# third party
import _pytest
from packaging import version
import pytest

# syft absolute
from syft.lib import vendor_requirements_available
from syft.lib import VendorLibraryImportException


def pytest_addoption(parser: _pytest.config.argparsing.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: _pytest.config.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "fast: mark test as fast to run")
    config.addinivalue_line("markers", "all: all tests")
    config.addinivalue_line("markers", "asyncio: mark test as asyncio")
    config.addinivalue_line("markers", "vendor: mark test as vendor library")
    config.addinivalue_line("markers", "libs: runs valid vendor tests")


def pytest_collection_modifyitems(
    config: _pytest.config.Config, items: TypeList[TypeAny]
) -> None:
    # $ pytest -m fast for the fast tests
    # $ pytest -m slow for the slow tests
    # $ pytest -m all for all the tests
    # $ pytest -m libs for the vendor tests
    slow_tests = pytest.mark.slow
    fast_tests = pytest.mark.fast
    all_tests = pytest.mark.all

    # dynamically filtered vendor lib tests
    # there isn't any way to remove "vendor" so the only way to filter
    # these tests is to add a different tag called "libs" and then run
    # the tests against that dynamic keyword
    vendor_tests = pytest.mark.libs  # note libs != vendor
    for item in items:
        # mark with: pytest.mark.vendor
        # run with: pytest -m libs -n auto 0
        if "vendor" in item.keywords:
            vendor_requirements = item.own_markers[0].kwargs
            try:
                if vendor_requirements_available(
                    vendor_requirements=vendor_requirements
                ):
                    item.add_marker(vendor_tests)
            except VendorLibraryImportException as e:
                print(e)
            except Exception as e:
                print(f"Unable to check vendor library: {vendor_requirements}. {e}")

        item.add_marker(all_tests)
        if "slow" in item.keywords:
            item.add_marker(slow_tests)
        else:
            item.add_marker(fast_tests)
