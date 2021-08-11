# stdlib
import operator
from typing import Any
from typing import Dict as TypeDict
from typing import Union

# third party
import numpy as np
import pytest
import torch
from typing_extensions import Final

# syft absolute
import syft as sy
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
from syft.core.tensor.smpc.share_tensor import ShareTensor

# TODO: The functionality for VMs was removed to focus on the Hagrid
vms = [sy.VirtualMachine(name=name) for name in ["alice", "bob", "theo", "andrew"]]
clients = [vm.get_client() for vm in vms]

PUBLIC_VALUES: Final[TypeDict[str, Union[np.ndarray, torch.Tensor, int]]] = {
    "numpy_array": np.array([32], dtype=np.int64),
    "torch_tensor": torch.tensor([42], dtype=torch.int64),
    "int": 42,
}


@pytest.mark.parametrize("public_value_type", ["int", "torch_tensor", "numpy_array"])
def test_remote_sharing(public_value_type: str) -> None:
    value = np.array([[1, 2, 3, 4, -5]], dtype=np.int64)
    remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

    mpc_tensor = MPCTensor(
        parties=clients, secret=remote_value, shape=(1, 5), seed_shares=52
    )

    assert len(mpc_tensor.child) == len(clients)

    shares = [share.get_copy() for share in mpc_tensor.child]
    assert all(isinstance(share, ShareTensor) for share in shares)
    assert (mpc_tensor.reconstruct() == value).all()


@pytest.mark.parametrize("op_str", ["add", "sub"])
def test_mpc_private_private_op(op_str: str) -> None:
    value_1 = np.array([[1, 2, 3, 4, -5]], dtype=np.int64)
    value_2 = np.array([42], dtype=np.int64)

    remote_value_1 = clients[0].syft.core.tensor.tensor.Tensor(value_1)
    remote_value_2 = clients[2].syft.core.tensor.tensor.Tensor(value_2)

    mpc_tensor_1 = MPCTensor(
        parties=clients, secret=remote_value_1, shape=(1, 5), seed_shares=52
    )

    mpc_tensor_2 = MPCTensor(
        parties=clients, secret=remote_value_2, shape=(1,), seed_shares=42
    )

    op = getattr(operator, op_str)

    res = op(mpc_tensor_1, mpc_tensor_2).reconstruct()
    expected = op(value_1, value_2)

    assert (res == expected).all()


@pytest.mark.parametrize("public_value_type", ["int", "torch_tensor", "numpy_array"])
@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_mpc_private_public_op(op_str: str, public_value_type: str) -> None:
    value_1 = np.array([[1, 2, 3, 4, -5]], dtype=np.int64)
    value_2 = PUBLIC_VALUES[public_value_type]

    remote_value_1 = clients[0].syft.core.tensor.tensor.Tensor(value_1)

    mpc_tensor_1 = MPCTensor(
        parties=clients, secret=remote_value_1, shape=(1, 5), seed_shares=52
    )

    op = getattr(operator, op_str)

    res = op(mpc_tensor_1, value_2).reconstruct()

    # TODO: Conversion to numpy is required because numpy op torch_tensor
    # gives back
    # TypeError: Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead.
    expected = op(value_1, np.array(value_2))

    assert (res == expected).all()


@pytest.mark.parametrize("public_value_type", ["int", "torch_tensor", "numpy_array"])
@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_mpc_public_private_op(op_str: str, public_value_type: str) -> None:
    value_1 = PUBLIC_VALUES[public_value_type]
    value_2 = np.array([[1, 2, 3, 4, -5]], dtype=np.int64)

    remote_value_2 = clients[0].syft.core.tensor.tensor.Tensor(value_2)

    mpc_tensor_2 = MPCTensor(
        parties=clients, secret=remote_value_2, shape=(1, 5), seed_shares=52
    )

    op = getattr(operator, op_str)

    # TODO: Conversion to numpy is required because numpy op torch_tensor
    # gives back
    # TypeError: Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead.
    res = op(value_1, mpc_tensor_2).reconstruct()
    expected = op(value_1, value_2)

    assert (res == np.array(expected)).all()


@pytest.mark.parametrize(
    "method_str, kwargs", [("sum", {"axis": 0}), ("sum", {"axis": 1})]
)
def test_mpc_forward_methods(method_str: str, kwargs: TypeDict[str, Any]) -> None:
    value = np.array([[1, 2, 3, 4, -5], [5, 6, 7, 8, 9]], dtype=np.int64)

    remote_value = clients[0].syft.core.tensor.tensor.Tensor(value)

    mpc_tensor = MPCTensor(
        parties=clients, secret=remote_value, shape=(2, 5), seed_shares=52
    )

    op_mpc = getattr(mpc_tensor, method_str)
    op = getattr(value, method_str)

    res = op_mpc(**kwargs).reconstruct()
    expected = op(**kwargs)

    assert (res == expected).all()
