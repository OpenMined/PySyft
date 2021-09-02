# stdlib
import time

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.tensor.smpc.mpc_tensor import MPCTensor

sy.logger.remove()

PORT = 9081

PARTIES = 2


@pytest.mark.integration
def test_secret_sharing() -> None:
    clients = []
    for i in range(PARTIES):
        client = sy.login(
            email="info@openmined.org", password="changethis", port=(PORT + i)
        )
        clients.append(client)

    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
    value_secret = clients[0].syft.core.tensor.tensor.Tensor(data)
    mpc_tensor = MPCTensor(secret=value_secret, shape=(2, 5), parties=clients)

    # wait for network comms between nodes
    time.sleep(2)

    res = mpc_tensor.reconstruct()
    assert (res == data).all()

    public_value = 42
    res_add = mpc_tensor + public_value

    # wait for network comms between nodes
    time.sleep(2)

    result = res_add.reconstruct()
    assert ((data + public_value) == result).all()

    res_sub = mpc_tensor - public_value

    # wait for network comms between nodes
    time.sleep(2)

    result = res_sub.reconstruct()
    assert ((data - public_value) == result).all()

    res_mul = mpc_tensor * public_value

    # wait for network comms between nodes
    time.sleep(2)

    result = res_mul.reconstruct()
    assert ((data * public_value) == result).all()
