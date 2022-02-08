# stdlib
from typing import Any

# third party
import numpy as np

# import pytest
import torch

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor
from syft.core.tensor.util import implements


def test_data_child() -> None:
    data = np.array([1, 2, 3], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert (tensor._data_child == data).all()


def test_len() -> None:
    data_list = [1.5, 3, True, "Thanos"]
    data_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
    for i in data_list:
        if i == float or int or bool:
            data = np.array([i])
            tensor = PassthroughTensor(child=data)

            assert tensor.__len__() == 1

        else:
            data = np.array([i])
            tensor = PassthroughTensor(child=data)

            assert len(tensor) == 1

    tensor = PassthroughTensor(child=data_array)

    assert len(tensor) == 2


def test_shape() -> None:
    data_list = [1.5, 3, True, "Thanos"]
    data_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
    for i in data_list:
        if i == float or int or bool:
            data = np.array([i])
            tensor = PassthroughTensor(child=data)

            assert tensor.shape == (1,)

        else:
            data = np.array([i])
            tensor = PassthroughTensor(child=data)

            assert tensor.shape == (1,)

    tensor = PassthroughTensor(child=data_array)

    assert tensor.shape == (2, 2)


def test_dtype() -> None:
    data = np.array([1, 2, 3], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.dtype == np.int32


# This test only works if the return statement for logical_and in passthrough.py
# is changed FROM:
#                       retrun self.__class__(self.child and other)
# TO:
#                       return self and other
#
# I could not figure out how to execute tests on the code the way it is now
# when trying to evaluate Passthrough tensors. The only way I got
# it to start producing the correct result was to change the return statement.
#
# Note: uncomment import pytest (line: 7) if running this test.

# def test_logical_and() -> None:
#     data_a = np.array([True, False, True])
#     data_b = np.array([False, False, True])
#     data_c = np.array([False, False])
#     tensor_a = PassthroughTensor(child=data_a)
#     tensor_b = PassthroughTensor(child=data_b)
#     tensor_c = PassthroughTensor(child=data_c)
#     expected = tensor_b
#     result = tensor_a.logical_and(tensor_b)

#     assert result == expected

#     with pytest.raises(Exception,
#         match = "Tensor shapes do not match for __eq__: [0-9]+ != [0-9]+"):
#             tensor_b.logical_and(tensor_c)


def test__abs__() -> None:
    data = np.array([1, -1, -2], dtype=np.int32)
    expected = np.array([1, 1, 2], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)

    assert tensor_a.__abs__() == tensor_b


def test__add__() -> None:
    data_a = np.array([1, -1, -2], dtype=np.int32)
    data_b = np.array([1, 1, 3], dtype=np.int32)
    expected = np.array([2, 0, 1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__add__(tensor_b)
    result_b = tensor_a.__add__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__radd__() -> None:
    data_a = np.array([1, -1, -2], dtype=np.int32)
    data_b = torch.tensor([1, 1, 3], dtype=torch.int32)
    expected = torch.tensor([2, 0, 1], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__radd__(tensor_b)
    result_b = tensor_a.__radd__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__sub__() -> None:
    data_a = np.array([1, -1, -2], dtype=np.int32)
    data_b = np.array([1, 1, 3], dtype=np.int32)
    expected = np.array([0, -2, -5], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__sub__(tensor_b)
    result_b = tensor_a.__sub__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__rsub__() -> None:
    data_a = np.array([1, -1, -2], dtype=np.int32)
    data_b = torch.tensor([1, 1, 3], dtype=torch.int32)
    expected = torch.tensor([0, -2, -5], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_b.__rsub__(tensor_a)
    result_b = tensor_b.__rsub__(data_a)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__gt__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = np.array([0, 3, 3], dtype=np.int32)
    expected = np.array([True, False, False])
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__gt__(tensor_b)
    result_b = tensor_a.__gt__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__ge__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = np.array([0, 3, 3], dtype=np.int32)
    expected = np.array([True, False, True])
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__ge__(tensor_b)
    result_b = tensor_a.__ge__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__lt__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = np.array([2, 1, 3], dtype=np.int32)
    expected = np.array([True, False, False])
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__lt__(tensor_b)
    result_b = tensor_a.__lt__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__le__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = np.array([0, 3, 3], dtype=np.int32)
    expected = np.array([False, True, True])
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__le__(tensor_b)
    result_b = tensor_a.__le__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__ne__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = np.zeros((3,), dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    result_a = tensor_a.__ne__(tensor_b)
    result_b = tensor_b.__ne__(data_b)

    assert result_a is True
    assert result_b is False


def test__eq__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = np.zeros((3,), dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    result_a = tensor_a.__eq__(tensor_b)
    result_b = tensor_a.__eq__(data_a)

    assert result_a is False
    assert result_b is True


def test__floordiv__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = np.array([1, 5, 4], dtype=np.int32)
    expected = np.array([1, 2, 0], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__floordiv__(tensor_b)
    result_b = tensor_a.__floordiv__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__rfloordiv__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = torch.tensor([1, 5, 4], dtype=torch.int32)
    expected = torch.tensor([1, 2, 0], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__rfloordiv__(tensor_b)
    result_b = tensor_a.__rfloordiv__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__lshift__() -> None:
    data_a = np.array([1, 2, -1], dtype=np.int32)
    data_b = np.array([0, 2, 1], dtype=np.int32)
    expected = np.array([0, 8, -2], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__lshift__(tensor_b)
    result_b = tensor_a.__lshift__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__rlshift__() -> None:
    data_a = np.array([1, 2, -1], dtype=np.int32)
    data_b = torch.tensor([0, 2, 1], dtype=torch.int32)
    expected = torch.tensor([0, 8, -2], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__rlshift__(tensor_b)
    result_b = tensor_a.__rlshift__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__rshift__() -> None:
    data_a = np.array([1, 2, -1], dtype=np.int32)
    data_b = np.array([2, 1, 1], dtype=np.int32)
    expected = np.array([0, 1, -1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__rshift__(tensor_b)
    result_b = tensor_a.__rshift__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__rrshift__() -> None:
    data_a = np.array([1, 2, -1], dtype=np.int32)
    data_b = torch.tensor([0, 2, 1], dtype=torch.int32)
    expected = torch.tensor([0, 8, -1], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__rrshift__(tensor_b)
    result_b = tensor_a.__rrshift__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__pow__() -> None:
    data_a = np.array([1, 2, -1], dtype=np.int32)
    data_b = np.array([0, 2, 1], dtype=np.int32)
    expected = np.array([1, 4, -1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__pow__(tensor_b)
    result_b = tensor_a.__pow__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__rpow__() -> None:
    data_a = np.array([1, 2, 1], dtype=np.int32)
    data_b = torch.tensor([0, 2, -1], dtype=torch.int32)
    expected = torch.tensor([0, 4, -1], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__rpow__(tensor_b)
    result_b = tensor_a.__rpow__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__divmod__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = np.array([1, 5, 4], dtype=np.int32)
    expected = np.array([0, 1, 3], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__divmod__(tensor_b)
    result_b = tensor_a.__divmod__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__neg__() -> None:
    data_a = np.array([1, 0, -1], dtype=np.int32)
    expected = np.array([-1, 0, 1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=expected)

    assert tensor_a.__neg__() == tensor_b


def test__invert__() -> None:
    data_a = np.array([-1, 0, 1], dtype=np.int32)
    expected = np.array([0, -1, -2], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=tensor_a.__invert__())
    tensor_c = PassthroughTensor(child=expected)

    assert tensor_b == tensor_c


def test__index__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)

    assert tensor_a[2].__index__() == 3


def test_copy() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)

    assert tensor_a.copy() == tensor_a


def test__mul__() -> None:
    data_a = np.array([1, 2, -1], dtype=np.int32)
    data_b = np.array([0, 2, 1], dtype=np.int32)
    expected = np.array([0, 4, -1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__mul__(tensor_b)
    result_b = tensor_a.__mul__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__rmul__() -> None:
    data_a = np.array([1, 2, -1], dtype=np.int32)
    data_b = torch.tensor([0, 2, 1], dtype=torch.int32)
    expected = torch.tensor([0, 4, -1], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__rmul__(tensor_b)
    result_b = tensor_a.__rmul__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__matmul__() -> None:
    data_a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    data_b = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected = np.array([[7, 10], [15, 22]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=expected)

    assert tensor_a.__matmul__(data_b) == tensor_b


def test__rmatmul__() -> None:
    data_a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    data_b = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    expected = np.array([[7, 10], [15, 22]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)

    assert tensor_a.__rmatmul__(tensor_b) == tensor_c


def test__truediv__() -> None:
    data_a = np.array([1, 2, -3], dtype=np.int32)
    data_b = np.array([1, 3, 2], dtype=np.int32)
    expected = np.array([1.0, 0.66666667, -1.5], dtype=np.float32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result_a = tensor_a.__truediv__(tensor_b)
    result_b = tensor_a.__truediv__(data_b)

    assert result_a == tensor_c
    assert result_b == tensor_c


def test__rtruediv__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    data_b = torch.tensor([1.0, 3.0, -3.0], dtype=torch.float32)
    expected = torch.tensor([1.0, 1.5, -1.0], dtype=torch.float32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=expected)
    result = tensor_a.__rtruediv__(tensor_b)

    assert result == tensor_c


def test_manual_dot() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected = np.array([[7, 10], [15, 22]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)

    assert tensor_a.manual_dot(data) == tensor_b


def test_dot() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected = np.array([[7, 10], [15, 22]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)

    assert tensor_a.dot(data) == tensor_b


def test_reshape() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected = np.array([1, 2, 3, 4], dtype=np.int32)
    tensor_a = PassthroughTensor(data)
    tensor_b = PassthroughTensor(expected)
    result = tensor_a.reshape((1, 4))

    assert result == tensor_b


def test_repeat() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected = np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.int32)
    tensor_a = PassthroughTensor(data)
    tensor_b = PassthroughTensor(expected)
    result = tensor_a.repeat(2, axis=1)

    assert result == tensor_b


def test_resize() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected = np.array([[1, 2, 3, 4, 0]], dtype=np.int32)
    tensor_a = PassthroughTensor(data)
    tensor_b = PassthroughTensor(expected)
    result = tensor_a.resize((5, 1))

    assert result == tensor_b


def test_T() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int32)
    tensor_a = PassthroughTensor(data)
    tensor_b = PassthroughTensor(expected)
    result = tensor_a.T

    assert result == tensor_b


def test_transpose() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int32)
    tensor_a = PassthroughTensor(data)
    tensor_b = PassthroughTensor(expected)
    result = tensor_a.transpose((1, 0))

    assert result == tensor_b


def test__getitem__() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected = np.array([3, 4], dtype=np.int32)
    index = 1
    tensor_a = PassthroughTensor(data)
    tensor_b = PassthroughTensor(expected)
    tensor_c = PassthroughTensor(index)
    result_a = tensor_a.__getitem__(index)
    result_b = tensor_a.__getitem__(tensor_c)

    assert result_a == tensor_b
    assert result_b == tensor_b


def test_argmax() -> None:
    data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    expected = np.array([1, 1, 1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.argmax(0)

    assert result == tensor_b


def test_argmin() -> None:
    data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    expected = np.array([0, 0, 0], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.argmin(0)

    assert result == tensor_b


def test_argsort() -> None:
    data = np.array([[1, 2, 0], [3, 4, 5]], dtype=np.int32)
    expected = np.array([2, 0, 1, 0, 1, 2], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.argsort(axis=None)

    assert result == tensor_b


def test_sort() -> None:
    data_a = np.array([[1, 4], [3, 1]], dtype=np.int32)
    expected = np.array([[1, 4], [1, 3]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=expected)
    tensor_a.sort(axis=0, kind="quicksort")

    assert tensor_a == tensor_b


def test_clip() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    expected = np.array([[3, 3, 3], [4, 4, 4]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.clip(3, 4)

    assert result == tensor_b


def test_cumprod() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    expected = np.array([1, 2, 6, 24, 120, 720], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.cumprod(axis=None)

    assert result == tensor_b


def test_cumsum() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    expected = np.array([1, 3, 6, 10, 15, 21], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.cumsum(axis=None)

    assert result == tensor_b


def test_trace() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    expected = np.array([6, 8], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.trace()

    assert result == tensor_b


def test_diagonal() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    expected = np.array([[0, 6], [1, 7]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.diagonal()

    assert result == tensor_b


def test_tolist() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    expected = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.tolist()

    assert result == tensor_b
    assert type(result.child) == list


def test_flatten() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    expected = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.flatten()

    assert result == tensor_b


def test_partition() -> None:
    data = np.array([3, 1, 2, 0, 7, 6, 4, 5], dtype=np.int32)
    expected = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.partition((1, 3, 5, 7))

    assert result == tensor_b


def test_ravel() -> None:
    data = np.array([[[2, 1], [0, 5]], [[7, 3], [6, 4]]], dtype=np.int32)
    expected = np.array([2, 1, 0, 5, 7, 3, 6, 4], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.ravel()

    assert result == tensor_b


def test_compress() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    expected = np.array([[[0], [2]], [[4], [6]]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.compress([True, False], axis=2)

    assert result == tensor_b


def test_swapaxes() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    expected = np.array([[[0, 4], [2, 6]], [[1, 5], [3, 7]]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.swapaxes(axis1=0, axis2=2)

    assert result == tensor_b


def test_put() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [5, 7]]], dtype=np.int32)
    expected = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = tensor_a.put(6, 6)

    assert result == tensor_b


def test__pos__() -> None:
    data = np.array([-1], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.__pos__().child == -1


def test_mean() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.mean().child == 3.5


def test_max() -> None:
    data = np.array([[[0, -1], [2, 3]], [[4, 7], [6, 5]]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.max().child == 7


def test_min() -> None:
    data = np.array([[[0, -1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.min().child == -1


def test_ndim() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.ndim == 3


def test_prod() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.prod().child == 24


def test_squeeze() -> None:
    data = np.array([[[0], [1], [2]]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)
    result = tensor.squeeze()

    assert result.shape == (3,)


def test_std() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.std().child.round(4) == 2.2913


# Needs an additional test for PassthroughTensor with "copy_tensor" attribute.
def test_sum() -> None:
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.sum().child == 28


def test_take() -> None:
    data_a = np.array([[[5, 15], [25, 35]], [[45, 55], [65, 75]]], dtype=np.int32)
    data_b = np.array([1, 2], dtype=np.int32)
    expected = np.array([15, 25], dtype=np.int32)
    tensor = PassthroughTensor(child=data_a)

    assert (tensor.take(data_b).child == expected).all()


def test_choose() -> None:
    choices = np.array(
        [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=np.int32
    )
    data_a = np.array([0, 0, 1])
    data_b = np.array([2, 0, 1])
    tensor_data_a = PassthroughTensor(child=data_a)
    tensor_data_b = PassthroughTensor(child=data_b)
    expected_a = np.array([[0, 1, 5], [4, 5, 11]], dtype=np.int32)
    expected_b = np.array([[6, 1, 8], [9, 4, 11]], dtype=np.int32)
    expected_c = np.array([[0, 1, 8], [3, 4, 11]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=expected_a)
    tensor_b = PassthroughTensor(child=expected_b)
    tensor_c = PassthroughTensor(child=expected_c)
    result_raise = tensor_data_a.choose(choices)
    result_clip = tensor_data_b.choose(choices, mode="clip")
    result_warp = tensor_data_b.choose(choices, mode="warp")

    assert result_raise == tensor_a
    assert result_clip == tensor_b
    assert result_warp == tensor_c


def test_astype() -> None:
    data = np.array([[-1.1, 2.2], [3.99, 4.0]], dtype=np.float32)
    expected = np.array([[-1, 2], [3, 4]], dtype=np.int32)
    tensor = PassthroughTensor(child=data)
    result = tensor.astype(int)

    assert (result.child == expected).all()


# Passthrough Subclass for testing __array_function__
class PtTensorSubclass(PassthroughTensor):
    def __init__(self, child: Any, unit=""):
        self.child = child
        self.unit = unit

    def __repr__(self):
        return f"PtTensorSubclass: {self.child}, {self.unit}"

    def __str__(self):
        return f"{self.child} {self.unit}"


def __add__(self, other):
    if self.unit == other.unit:
        result = np.subtract(self.child + other.child)
        return PtTensorSubclass(result, self.unit)
    else:
        raise ValueError("PtTensorSubclass(for testing) must have the same units")


@implements(PtTensorSubclass, np.subtract)
def __sub__for_PtTensorSubclass(self, other) -> PtTensorSubclass:
    if self.unit == other.unit:
        result = self.child - other.child
        return PtTensorSubclass(result, self.unit)
    else:
        raise ValueError("PtTensorSubclass(for testing) must have the same units")


@implements(PtTensorSubclass, np.mean)
def np_mean_for_PtTensorSubclass(self, *args, **kwargs):
    mean_value = np.mean(self.child, *args, **kwargs)
    return self.__class__(mean_value, self.unit)


def test__array_function__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)

    # Testing NotImplemented
    tensor_a = PassthroughTensor(child=data_a)
    result_a = tensor_a.__array_function__(
        func=PtTensorSubclass.__add__,
        types=[PassthroughTensor, np.ndarray],
        args="num1",
        kwargs={},
    )

    assert result_a == NotImplemented

    # Testing implementation True
    tensor_b = PtTensorSubclass(child=data_a, unit="Hedgehogs")
    tensor_b.__array_function__(
        func=np_mean_for_PtTensorSubclass,
        types=[PtTensorSubclass],
        args=[tensor_b],
        kwargs={},
    )
    result_b = np.mean(tensor_b)
    expected_b = PtTensorSubclass(child=[2.0], unit="Hedgehogs")

    assert result_b == expected_b

    # Testing implementation False
    result_c = tensor_b.__array_function__(
        func=PtTensorSubclass.__add__,
        types=[PtTensorSubclass],
        args=[tensor_b, tensor_b],
        kwargs={},
    )
    data_b = np.array([2, 4, 6], dtype=np.int32)
    expected_c = PtTensorSubclass(child=data_b, unit="Hedgehogs")

    assert result_c == expected_c.child


def test__array_ufunc__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)

    # Testing implementation False
    data_b = np.array([2, 4, 6], dtype=np.int32)
    tensor_a = PtTensorSubclass(child=data_a, unit="Hedgehogs")
    expected_a = PtTensorSubclass(child=data_b, unit="Hedgehogs")
    result_a = tensor_a.__array_ufunc__(
        PtTensorSubclass.__add__, "__call__", tensor_a, tensor_a
    )

    assert result_a == expected_a

    # Testing implementation True
    data_c = np.array([0, 0, 0], dtype=np.int32)
    expected_b = PtTensorSubclass(child=data_c, unit="Hegehogs")
    result_b = tensor_a.__array_ufunc__(np.subtract, "__call__", tensor_a, tensor_a)

    assert result_b == expected_b


def test_repr() -> None:
    data = np.array([0, 1, 2, 3], dtype=np.int32)
    tensor = PassthroughTensor(child=data)
    result = tensor.__repr__()

    assert result == "PassthroughTensor(child=[0 1 2 3])"
    assert type(result) == str


def test_square() -> None:
    data = np.array([[0, 1], [-2, 3]], dtype=np.int32)
    expected = np.array([[0, 1], [4, 9]], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=expected)
    result = np.square(tensor_a)

    assert result == tensor_b
