# stdlib
import operator
from typing import Any
from typing import List

# third party
import pytest
import torch

# syft absolute
from syft.lib.tensor.tensorbase import DataTensor
from syft.lib.tensor.tensorbase import FloatTensor
from syft.lib.tensor.tensorbase import SyftTensor
from syft.lib.tensor.tensorbase import TENSOR_FORWARD_PROPERTIES


def get_children_types(t: Any, types: Any = None) -> List[Any]:
    types = [] if types is None else types
    if hasattr(t, "child"):
        return types + [type(t)] + get_children_types(t.child, types)
    else:
        return types + [type(t)]


def test_children() -> None:
    t = SyftTensor.FloatTensor([1, 2, 3])
    assert get_children_types(t) == [SyftTensor, FloatTensor, DataTensor, torch.Tensor]


def test_addition_floattensor() -> None:
    t1 = SyftTensor.FloatTensor([1, 2, 3])
    t2 = SyftTensor.FloatTensor([4, 5, 6])
    t3 = t1 + t2
    assert isinstance(t3, SyftTensor)
    assert torch.equal(t3.data, torch.tensor([5.0, 7.0, 9.0]))


def test_subtraction_floattensor() -> None:
    t1 = SyftTensor.FloatTensor([1, 2, 3])
    t2 = SyftTensor.FloatTensor([4, 5, 6])
    t3 = t1 - t2
    assert isinstance(t3, SyftTensor)
    assert torch.equal(t3.data, torch.tensor([-3.0, -3.0, -3.0]))


def test_multiplication_floattensor() -> None:
    t1 = SyftTensor.FloatTensor([1, 2, 3])
    t2 = SyftTensor.FloatTensor([4, 5, 6])
    t3 = t1 * t2
    assert torch.equal(t3.data, torch.tensor([4.0, 10.0, 18.0]))


def test_division_floattensor() -> None:
    t1 = SyftTensor.FloatTensor([6, 9, 12])
    integer = 3
    t3 = t1 / integer
    assert torch.equal(t3.data, torch.tensor([2.0, 3.0, 4.0]))


def test_matmul_floattensor() -> None:
    t1 = SyftTensor.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t2 = SyftTensor.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    t3 = t1 @ t2
    assert torch.equal(t3.data, torch.tensor([[22.0, 28.0], [49.0, 64.0]]))


def test_matmul_inttensor() -> None:
    t1 = SyftTensor.IntegerTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t2 = SyftTensor.IntegerTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    t3 = t1 @ t2
    assert torch.equal(t3.data, torch.IntTensor([[22, 28], [49, 64]]))


@pytest.mark.parametrize("op_str", ["matmul", "truediv", "add", "sub"])
def test_operation_exception(op_str: str) -> None:
    t1 = SyftTensor.IntegerTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t2 = torch.Tensor([1, 2, 3])

    op = getattr(operator, op_str)
    with pytest.raises(ValueError):
        op(t1, t2)


def test_delegation_transpose() -> None:
    # this delegates transpose down to the lowest child,
    # and wraps the result using the same class tree as the original object
    t = SyftTensor.FloatTensor([[4, 5, 6]])
    t_t = t.transpose(0, 1)

    assert get_children_types(t) == [SyftTensor, FloatTensor, DataTensor, torch.Tensor]
    assert get_children_types(t_t) == [
        SyftTensor,
        FloatTensor,
        DataTensor,
        torch.Tensor,
    ]
    assert isinstance(t_t, SyftTensor)
    assert all(t_t.data == torch.tensor([[4.0], [5.0], [6.0]]))


@pytest.mark.parametrize("property_name", TENSOR_FORWARD_PROPERTIES)
def test_get_properties(property_name: str) -> None:
    tensor = torch.tensor([[4, 5, 6], [1, 2, 3]])
    t = SyftTensor.FloatTensor([[4, 5, 6], [1, 2, 3]])

    res = getattr(t, property_name)
    res_expected = getattr(tensor, property_name)

    if isinstance(res, torch.Tensor):
        assert torch.all(res == res_expected)
    else:
        assert res == res_expected


def test_exception_method() -> None:
    t = SyftTensor.FloatTensor([[4, 5, 6], [1, 2, 3]])

    with pytest.raises(AttributeError):
        getattr(t, "size")
