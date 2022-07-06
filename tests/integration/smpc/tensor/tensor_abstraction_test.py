"""Methods to test Tensor abstraction of MPCTensor"""

# stdlib
import operator

# third party
import numpy as np
import pytest

# syft absolute
from syft import Tensor
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE


@pytest.mark.smpc
@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_tensor_abstraction_pointer(get_clients, op_str) -> None:
    clients = get_clients(3)

    op = getattr(operator, op_str)

    data_1 = Tensor(child=np.array([[15, 34], [32, 89]], dtype=DEFAULT_INT_NUMPY_TYPE))
    data_2 = Tensor(child=np.array([[567, 98], [78, 25]], dtype=DEFAULT_INT_NUMPY_TYPE))
    data_3 = Tensor(
        child=np.array([[125, 10], [124, 28]], dtype=DEFAULT_INT_NUMPY_TYPE)
    )

    tensor_pointer_1 = data_1.send(clients[0])
    tensor_pointer_2 = data_2.send(clients[1])
    tensor_pointer_3 = data_3.send(clients[2])

    # creates an MPCTensor between party 1 and party 2
    mpc_1_2 = op(tensor_pointer_1, tensor_pointer_2)
    mpc_1_2.block_with_timeout(secs=40)

    # creates and MPCTensor between party 1,2,3
    mpc_1_2_3 = op(mpc_1_2, tensor_pointer_3)
    mpc_1_2_3.block_with_timeout(secs=40)

    exp_res = op(data_1, data_2)

    assert (mpc_1_2.reconstruct() == exp_res.child).all()

    exp_res = op(exp_res, data_3)

    assert (mpc_1_2_3.reconstruct() == exp_res.child).all()


@pytest.mark.smpc
@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_tensor_abstraction_subsets(get_clients, op_str) -> None:
    clients = get_clients(3)

    op = getattr(operator, op_str)

    data_1 = Tensor(child=np.array([[15, 34], [32, 89]], dtype=DEFAULT_INT_NUMPY_TYPE))
    data_2 = Tensor(child=np.array([[567, 98], [78, 25]], dtype=DEFAULT_INT_NUMPY_TYPE))
    data_3 = Tensor(
        child=np.array([[125, 10], [124, 28]], dtype=DEFAULT_INT_NUMPY_TYPE)
    )

    tensor_pointer_1 = data_1.send(clients[0])
    tensor_pointer_2 = data_2.send(clients[1])
    tensor_pointer_3 = data_3.send(clients[2])

    # Tensor abstraction among different subsets of parties

    # creates an MPCTensor between party 1 and party 2
    mpc_1_2 = op(tensor_pointer_1, tensor_pointer_2)

    mpc_1_2.block_with_timeout(40)

    # creates and MPCTensor between party 2,3
    mpc_2_3 = op(tensor_pointer_2, tensor_pointer_3)
    mpc_2_3.block_with_timeout(40)

    # creates and MPCTensor between party 1,2,3
    mpc_1_2_3 = op(mpc_1_2, mpc_2_3)
    mpc_1_2_3.block_with_timeout(secs=40)

    exp_res_1 = op(data_1, data_2)
    assert (mpc_1_2.reconstruct() == exp_res_1.child).all()

    exp_res_2 = op(data_2, data_3)
    assert (mpc_2_3.reconstruct() == exp_res_2.child).all()

    exp_res_3 = op(exp_res_1, exp_res_2)
    assert (mpc_1_2_3.reconstruct() == exp_res_3.child).all()


@pytest.mark.smpc
def test_tensor_abstraction_concat(get_clients) -> None:
    clients = get_clients(3)

    data_1 = Tensor(child=np.array([[15, 34], [32, 89]], dtype=DEFAULT_INT_NUMPY_TYPE))
    data_2 = Tensor(child=np.array([[567, 98], [78, 25]], dtype=DEFAULT_INT_NUMPY_TYPE))
    data_3 = Tensor(
        child=np.array([[125, 10], [124, 28]], dtype=DEFAULT_INT_NUMPY_TYPE)
    )

    tensor_pointer_1 = data_1.send(clients[0])
    tensor_pointer_2 = data_2.send(clients[1])
    tensor_pointer_3 = data_3.send(clients[2])

    # Tensor abstraction among different subsets of parties

    # creates an MPCTensor between party 1 and party 2
    mpc_1_2 = tensor_pointer_1.concatenate(tensor_pointer_2)

    mpc_1_2.block_with_timeout(40)

    # creates and MPCTensor between party 2,3
    mpc_2_3 = tensor_pointer_2.concatenate(tensor_pointer_3)
    mpc_2_3.block_with_timeout(40)

    # creates and MPCTensor between party 1,2,3
    mpc_1_2_3 = mpc_1_2.concatenate(mpc_2_3)
    mpc_1_2_3.block_with_timeout(secs=40)

    exp_res_1 = np.concatenate((data_1.child, data_2.child))
    assert (mpc_1_2.reconstruct() == exp_res_1).all()

    exp_res_2 = np.concatenate((data_2.child, data_3.child))
    assert (mpc_2_3.reconstruct() == exp_res_2).all()

    exp_res_3 = np.concatenate((exp_res_1, exp_res_2))
    assert (mpc_1_2_3.reconstruct() == exp_res_3).all()
