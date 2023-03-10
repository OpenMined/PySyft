# third party
import _pytest


def pytest_configure(config: _pytest.config.Config) -> None:
    config.addinivalue_line("markers", "frontend: frontend integration tests")
    config.addinivalue_line("markers", "network: network integration tests")
