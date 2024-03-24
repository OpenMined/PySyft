# stdlib
from secrets import token_hex

# third party
import pytest

# syft absolute
import syft as sy
from syft.service.response import SyftError


@pytest.mark.local_node
def test_enclave_root_client_exception():
    enclave_node = sy.orchestra.launch(
        name=token_hex(8),
        node_type=sy.NodeType.ENCLAVE,
        dev_mode=True,
        reset=True,
        local_db=True,
    )
    res = enclave_node.login(email="info@openmined.org", password="changethis")
    assert isinstance(res, SyftError)
    enclave_node.python_node.cleanup()
    enclave_node.land()
