# stdlib
import operator
import time

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.tensor import Tensor
from syft.core.tensor.smpc.mpc_tensor import MPCTensor

sy.logger.remove()


@pytest.mark.integration
def test_secret_sharing(get_clients) -> None:
    clients = get_clients(2)

    data = Tensor(child=np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32))
    value_secret = data.send(clients[0])

    mpc_tensor = MPCTensor(secret=value_secret, shape=(2, 5), parties=clients)

    time.sleep(10)

    res = mpc_tensor.reconstruct()
    assert (res == data.child).all()


@pytest.mark.integration
@pytest.mark.parametrize("op_str", ["add", "sub"])
def test_mpc_private_private_op(get_clients, op_str: str) -> None:
    clients = get_clients(2)

    value_1 = Tensor(child=np.array([[1, 2, 3, 4, -5]], dtype=np.int32))
    value_2 = Tensor(child=np.array([42], dtype=np.int32))

    remote_value_1 = value_1.send(clients[0])
    remote_value_2 = value_2.send(clients[1])

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(1, 5))
    mpc_tensor_2 = MPCTensor(parties=clients, secret=remote_value_2, shape=(1,))

    op = getattr(operator, op_str)
    res_ptr = op(mpc_tensor_1, mpc_tensor_2)

    time.sleep(20)

    res = res_ptr.reconstruct()
    expected = op(value_1, value_2)

    assert (res == expected.child).all()


@pytest.mark.integration
@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_mpc_public_private_op(get_clients, op_str: str) -> None:
    clients = get_clients(2)

    value_1 = Tensor(child=np.array([[1, 2, 3, 4, -5]], dtype=np.int32))

    remote_value_1 = value_1.send(clients[0])
    public_value = 27

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(1, 5))

    op = getattr(operator, op_str)

    res = op(mpc_tensor_1, public_value)

    time.sleep(20)

    res = res.reconstruct()
    expected = op(value_1, public_value)

    assert (res == expected.child).all()
