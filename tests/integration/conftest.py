# third party
import _pytest
import pytest


def pytest_configure(config: _pytest.config.Config) -> None:
    config.addinivalue_line("markers", "frontend: frontend integration tests")
    config.addinivalue_line("markers", "network: network integration tests")


@pytest.fixture
def gateway_port() -> int:
    return 9081


@pytest.fixture
def domain_1_port() -> int:
    return 9082


@pytest.fixture
def domain_2_port() -> int:
    return 9083
