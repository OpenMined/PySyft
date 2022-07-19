# stdlib
import operator
from typing import Any
from typing import Dict as TypeDict

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft import Tensor
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
from syft.core.tensor.smpc.mpc_tensor import ShareTensor

sy.logger.remove()


@pytest.mark.smpc
def test_secret_sharing(get_clients) -> None:
    clients = get_clients(2)

    data = Tensor(
        child=np.array(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=DEFAULT_INT_NUMPY_TYPE
        )
    )
    value_secret = data.send(clients[0])

    mpc_tensor = MPCTensor(secret=value_secret, shape=(2, 5), parties=clients)

    mpc_tensor.block_with_timeout(secs=20)

    assert len(mpc_tensor.child) == len(clients)

    shares = [share.get_copy() for share in mpc_tensor.child]
    assert all(isinstance(share, Tensor) for share in shares)
    assert all(isinstance(share.child, ShareTensor) for share in shares)

    res = mpc_tensor.reconstruct()
    assert (res == data.child).all()


@pytest.mark.smpc
@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_mpc_private_private_op(get_clients, op_str: str) -> None:
    clients = get_clients(2)

    value_1 = Tensor(child=np.array([[1, 2, 3, 4, -5]], dtype=DEFAULT_INT_NUMPY_TYPE))
    value_2 = Tensor(child=np.array([42], dtype=DEFAULT_INT_NUMPY_TYPE))

    remote_value_1 = value_1.send(clients[0])
    remote_value_2 = value_2.send(clients[1])

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(1, 5))
    mpc_tensor_2 = MPCTensor(parties=clients, secret=remote_value_2, shape=(1,))

    op = getattr(operator, op_str)
    res_ptr = op(mpc_tensor_1, mpc_tensor_2)

    res_ptr.block_with_timeout(secs=40)

    res = res_ptr.reconstruct()
    expected = op(value_1, value_2)

    assert (res == expected.child).all()


@pytest.mark.smpc
@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_mpc_private_public_op(get_clients, op_str: str) -> None:
    clients = get_clients(2)

    value_1 = Tensor(child=np.array([[1, 2, 3, 4, -5]], dtype=DEFAULT_INT_NUMPY_TYPE))

    remote_value_1 = value_1.send(clients[0])
    public_value = 27

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(1, 5))

    op = getattr(operator, op_str)

    res = op(mpc_tensor_1, public_value)
    res.block_with_timeout(secs=20)

    res = res.reconstruct()
    expected = op(value_1, public_value)

    assert (res == expected.child).all()


# TODO: Rasswanth to fix later after Tensor matmul refactor
@pytest.mark.skip
@pytest.mark.smpc
@pytest.mark.parametrize("op_str", ["matmul"])
def test_mpc_matmul_public(get_clients, op_str: str) -> None:
    clients = get_clients(2)

    value_1 = np.array([[1, 7], [3, -7]], dtype=DEFAULT_INT_NUMPY_TYPE)
    value_2 = np.array([[6, 2], [-6, 5]], dtype=DEFAULT_INT_NUMPY_TYPE)

    remote_value_1 = clients[0].syft.core.tensor.tensor.Tensor(value_1)

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(2, 2))

    op = getattr(operator, op_str)
    res = op(mpc_tensor_1, value_2)
    res.block_with_timeout(secs=80)

    res = res.reconstruct()

    expected = op(value_1, value_2)

    assert (res == expected).all()


@pytest.mark.smpc
@pytest.mark.parametrize(
    "method_str, kwargs", [("sum", {"axis": 0}), ("sum", {"axis": 1})]
)
def test_mpc_forward_methods(
    get_clients, method_str: str, kwargs: TypeDict[str, Any]
) -> None:
    clients = get_clients(2)
    value = np.array([[1, 2, 3, 4, -5], [5, 6, 7, 8, 9]], dtype=DEFAULT_INT_NUMPY_TYPE)

    remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

    mpc_tensor = MPCTensor(parties=clients, secret=remote_value, shape=(2, 5))

    op_mpc = getattr(mpc_tensor, method_str)
    op = getattr(value, method_str)

    res = op_mpc(**kwargs)

    res.block_with_timeout(secs=20)
    res = res.reconstruct()

    expected = op(**kwargs)

    assert (res == expected).all()


@pytest.mark.smpc_1
@pytest.mark.parametrize("op_str", ["lt", "gt", "ge", "le", "eq", "ne"])
def test_comp_mpc_private_private_op(get_clients, op_str: str) -> None:
    clients = get_clients(2)

    value_1 = Tensor(child=np.array([[1, 2, 3, 4, -5]], dtype=DEFAULT_INT_NUMPY_TYPE))
    value_2 = Tensor(child=np.array([0, 3, 98, -32, 27], dtype=DEFAULT_INT_NUMPY_TYPE))

    remote_value_1 = value_1.send(clients[0])
    remote_value_2 = value_2.send(clients[1])

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(1, 5))
    mpc_tensor_2 = MPCTensor(parties=clients, secret=remote_value_2, shape=(1, 5))

    op = getattr(operator, op_str)
    res_ptr = op(mpc_tensor_1, mpc_tensor_2)

    res_ptr.block_with_timeout(secs=120)

    res = res_ptr.reconstruct()
    expected = op(value_1, value_2)

    assert (res == expected.child).all()


@pytest.mark.smpc_1
@pytest.mark.parametrize("op_str", ["lt", "gt", "ge", "le", "eq", "ne"])
def test_comp_mpc_private_public_op(get_clients, op_str: str) -> None:
    clients = get_clients(2)

    value_1 = Tensor(
        child=np.array([[32, -27, 108, 1, 32]], dtype=DEFAULT_INT_NUMPY_TYPE)
    )

    remote_value_1 = value_1.send(clients[0])
    public_value = 27

    mpc_tensor_1 = MPCTensor(parties=clients, secret=remote_value_1, shape=(1, 5))

    op = getattr(operator, op_str)
    res_ptr = op(mpc_tensor_1, public_value)

    res_ptr.block_with_timeout(secs=120)

    res = res_ptr.reconstruct()
    expected = op(value_1, public_value)

    assert (res == expected.child).all()
