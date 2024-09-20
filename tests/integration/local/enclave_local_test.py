# stdlib
from secrets import token_hex

# third party
import pytest

# syft absolute
import syft as sy
from syft.types.errors import SyftException


@pytest.mark.local_server
def test_enclave_root_client_exception():
    enclave_server = sy.orchestra.launch(
        name=token_hex(8),
        server_type=sy.ServerType.ENCLAVE,
        dev_mode=True,
        reset=True,
    )
    with pytest.raises(SyftException):
        enclave_server.login(email="info@openmined.org", password="changethis")
    enclave_server.python_server.cleanup()
    enclave_server.land()
