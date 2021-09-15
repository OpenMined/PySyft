# stdlib
import operator
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
clients = []
for i in range(PARTIES):
    client = sy.login(
        email="info@openmined.org", password="changethis", port=(PORT + i)
    )
    clients.append(client)


@pytest.mark.integration
def test_secret_sharing() -> None:
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
    value_secret = clients[0].syft.core.tensor.tensor.Tensor(data)
    mpc_tensor = MPCTensor(secret=value_secret, shape=(2, 5), parties=clients)

    # wait for network comms between nodes
    time.sleep(2)

    res = mpc_tensor.reconstruct()
    assert (res == data).all()


@pytest.mark.integration
@pytest.mark.parametrize("op_str", ["add", "sub"])
def test_mpc_private_private_op(op_str: str) -> None:
    value_1 = np.array([[1, 2, 3, 4, -5]], dtype=np.int64)
    value_2 = np.array([42], dtype=np.int64)

    remote_value_1 = clients[0].syft.core.tensor.tensor.Tensor(value_1)
    remote_value_2 = clients[1].syft.core.tensor.tensor.Tensor(value_2)

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(1, 5))

    mpc_tensor_2 = MPCTensor(parties=clients, secret=remote_value_2, shape=(1,))

    op = getattr(operator, op_str)

    res = op(mpc_tensor_1, mpc_tensor_2)
    time.sleep(2)
    res = res.reconstruct()
    expected = op(value_1, value_2)

    assert (res == expected).all()


@pytest.mark.integration
@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_mpc_public_private_op(op_str: str) -> None:
    value_1 = np.array([[1, 2, 3, 4, -5]], dtype=np.int64)

    remote_value_1 = clients[0].syft.core.tensor.tensor.Tensor(value_1)

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(1, 5))

    public_value = 42

    op = getattr(operator, op_str)

    res = op(mpc_tensor_1, public_value)
    time.sleep(2)
    res = res.reconstruct()
    expected = op(value_1, public_value)

    assert (res == expected).all()
