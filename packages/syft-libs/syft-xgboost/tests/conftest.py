# third party
import pytest

# syft absolute
# absolute
import syft as sy
from syft import logger

# from .syft.notebooks import free_port

logger.remove()


@pytest.fixture(scope="session")
def node() -> sy.VirtualMachine:
    return sy.VirtualMachine(name="Bob")


@pytest.fixture(scope="session")
def client(node: sy.VirtualMachine) -> sy.VirtualMachineClient:
    return node.get_client()


@pytest.fixture(scope="session")
def root_client(node: sy.VirtualMachine) -> sy.VirtualMachineClient:
    return node.get_root_client()
