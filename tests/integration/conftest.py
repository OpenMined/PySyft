# stdlib
from secrets import token_hex

# third party
import _pytest
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.abstract_node import NodeSideType
from syft.node.worker import Worker


def pytest_configure(config: _pytest.config.Config) -> None:
    config.addinivalue_line("markers", "frontend: frontend integration tests")
    config.addinivalue_line("markers", "network: network integration tests")
    config.addinivalue_line(
        "markers", "container_workload: container workload integration tests"
    )
    config.addinivalue_line("markers", "local_node: local node integration tests")


@pytest.fixture
def gateway_port() -> int:
    return 9081


@pytest.fixture
def domain_1_port() -> int:
    return 9082


@pytest.fixture
def domain_2_port() -> int:
    return 9083


@pytest.fixture
def faker():
    return Faker()


@pytest.fixture(scope="function")
def full_low_worker(n_consumers: int = 3, create_producer: bool = True) -> Worker:
    _node = sy.orchestra.launch(
        node_side_type=NodeSideType.LOW_SIDE,
        name=token_hex(8),
        # dev_mode=True,
        reset=True,
        n_consumers=n_consumers,
        create_producer=create_producer,
        queue_port=None,
        local_db=False,
        thread_workers=False,
    )
    # startup code here
    yield _node
    # # Cleanup code
    _node.python_node.cleanup()
    _node.land()


@pytest.fixture(scope="function")
def full_high_worker(n_consumers: int = 3, create_producer: bool = True) -> Worker:
    _node = sy.orchestra.launch(
        node_side_type=NodeSideType.HIGH_SIDE,
        name=token_hex(8),
        # dev_mode=True,
        reset=True,
        n_consumers=n_consumers,
        create_producer=create_producer,
        queue_port=None,
        local_db=False,
        thread_workers=False,
    )
    # startup code here
    yield _node
    # Cleanup code
    _node.python_node.cleanup()
    _node.land()
