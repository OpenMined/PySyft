# third party
import pytest
import sys

pytest.importorskip("sympc")


@pytest.fixture(scope="session", autouse="True")
def check_python_version() -> None:
    if sys.version_info < (3, 7):
        pytest.skip("Python >= 3.7 required")
