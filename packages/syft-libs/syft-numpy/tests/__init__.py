# third party
import pytest

# syft absolute
import syft as sy


@pytest.fixture(scope="module")
def numpy_root_client(root_client: sy.VirtualMachineClient) -> sy.VirtualMachineClient:
    return root_client
