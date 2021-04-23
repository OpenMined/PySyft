# stdlib
import logging
from multiprocessing import Process
import socket
from time import time
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Generator
from typing import List as TypeList

# third party
import _pytest
import pytest

# syft absolute
import syft as sy
from syft import logger
from syft.grid.example_nodes.network import signaling_server as start_signaling_server
from syft.lib import VendorLibraryImportException
from syft.lib import _load_lib
from syft.lib import vendor_requirements_available

# syft relative
from .syft.notebooks import free_port

logger.remove()


@pytest.fixture(scope="session")
def signaling_server() -> Generator:
    port = free_port()
    proc = Process(target=start_signaling_server, args=(port, "127.0.0.1"))

    proc.start()
    start = time()

    while time() - start < 15:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
    else:
        raise TimeoutError("Can't connect to the signaling server")

    yield port

    proc.terminate()


@pytest.fixture
def caplog(caplog: _pytest.logging.LogCaptureFixture) -> Generator:
    class PropogateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    logger.add(PropogateHandler())
    yield caplog
    logger.remove()


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
    config.addinivalue_line("markers", "benchmark: runs benchmark tests")
    config.addinivalue_line("markers", "torch: runs torch tests")
    config.addinivalue_line("markers", "duet: runs duet notebook integration tests")
    config.addinivalue_line("markers", "grid: runs grid tests")


def pytest_collection_modifyitems(
    config: _pytest.config.Config, items: TypeList[TypeAny]
) -> None:
    # $ pytest -m fast for the fast tests
    # $ pytest -m slow for the slow tests
    # $ pytest -m all for all the tests
    # $ pytest -m libs for the vendor tests

    slow_tests = pytest.mark.slow
    fast_tests = pytest.mark.fast
    duet_tests = pytest.mark.duet
    grid_tests = pytest.mark.grid
    all_tests = pytest.mark.all

    # dynamically filtered vendor lib tests
    # there isn't any way to remove "vendor" so the only way to filter
    # these tests is to add a different tag called "libs" and then run
    # the tests against that dynamic keyword
    vendor_tests = pytest.mark.libs  # note libs != vendor
    loaded_libs: TypeDict[str, bool] = {}
    vendor_skip = pytest.mark.skip(reason="vendor requirements not met")
    for item in items:
        if item.location[0].startswith("PyGrid"):
            # Ignore if PyGrid folder checked out in main dir
            continue

        if "grid" in item.keywords:
            item.add_marker(grid_tests)
            continue
        # mark with: pytest.mark.vendor
        # run with: pytest -m libs -n auto 0
        if "vendor" in item.keywords:
            vendor_requirements = item.own_markers[0].kwargs

            # try to load the lib first and if it fails just skip
            if "lib" in vendor_requirements:
                lib_name = vendor_requirements["lib"]
                if lib_name not in loaded_libs:
                    try:
                        _load_lib(lib=lib_name)
                        loaded_libs[lib_name] = True
                    except Exception as e:
                        print(f"Failed to load {lib_name}. {e}")
                        loaded_libs[lib_name] = False
                if not loaded_libs[lib_name]:
                    item.add_marker(vendor_skip)
                    continue

            try:
                # test the vendor requirements of the specific test if the library
                # was loaded successfully
                if vendor_requirements_available(
                    vendor_requirements=vendor_requirements
                ):
                    if item.location[0].startswith("tests/syft/notebooks"):
                        item.add_marker(duet_tests)
                    else:
                        item.add_marker(vendor_tests)
                    item.add_marker(all_tests)

            except VendorLibraryImportException as e:
                print(e)
            except Exception as e:
                print(f"Unable to check vendor library: {vendor_requirements}. {e}")
            continue

        if "benchmark" in item.keywords:
            continue

        if "torch" in item.keywords:
            item.add_marker(all_tests)
            continue

        item.add_marker(all_tests)
        if "slow" in item.keywords:
            item.add_marker(slow_tests)
        else:
            if item.location[0].startswith("tests/syft/notebooks"):
                item.add_marker(duet_tests)
                continue
            # fast is the default catch all
            item.add_marker(fast_tests)


@pytest.fixture(scope="session")
def node() -> sy.VirtualMachine:
    return sy.VirtualMachine(name="Bob")


@pytest.fixture(autouse=True)
def node_store(node: sy.VirtualMachine) -> None:
    node.store.clear()


@pytest.fixture(scope="session")
def client(node: sy.VirtualMachine) -> sy.VirtualMachineClient:
    return node.get_client()


@pytest.fixture(scope="session")
def root_client(node: sy.VirtualMachine) -> sy.VirtualMachineClient:
    return node.get_root_client()
