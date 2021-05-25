import pytest
from syft.core.tensor.tensor import Tensor
import numpy as np
import torch as th
from typing import Tuple
from inspect import isdatadescriptor, isgetsetdescriptor

other = type("Other", tuple(), {})
unknown = type("Unknown", tuple(), {})

@pytest.fixture(scope="module")
def numpy_tensor_pair() -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    return (a, np.array(np.flip(a)))

@pytest.fixture(scope="module")
def torch_tensor_pair(numpy_tensor_pair: Tuple[np.ndarray, np.ndarray]) -> Tuple[th.Tensor,
                                                                                 th.Tensor]:
    return (th.tensor(numpy_tensor_pair[0]), th.tensor(numpy_tensor_pair[1]))


@pytest.fixture(scope="module")
def syft_tensor_pair(numpy_tensor_pair: Tuple[np.ndarray, np.ndarray]) -> Tuple[Tensor, Tensor]:
    return (Tensor(numpy_tensor_pair[0]).autograd(requires_grad=True), Tensor(numpy_tensor_pair[
                                                                                  1]).autograd(requires_grad=True))


@pytest.fixture(scope="function")
def tensor_pairs(syft_tensor_pair, numpy_tensor_pair, torch_tensor_pair):
    return (syft_tensor_pair, numpy_tensor_pair, torch_tensor_pair)


# Example on how to add tests:
# (["fcall_name_if_it's_uniform"], None <- it's a fcall with no arguments or a property)
# (["fcall_syft", "fcall_numpy", "fcall_torch"], [(0, ), {"data": None}]) <- a list with multiple
# input scenarios, namely:
# 1. you can pass a tuple specifying the *args
# 2. you can pass a dict specifying the **kwargs

test_methods = [
    (["__abs__"], None),
    (["__add__"], [(other,)]),
    (["__divmod__", "__divmod__", None], [(other,)]),
    (["__eq__"], [(other,)]),
    (["__floordiv__"], [(other,)]),
    (["__ge__"], [(other,)]),
    (["__getitem__"], [(0,)]),
    (["__gt__"], [(other,)]),
    (["__index__"], unknown),
    (["__invert__"], unknown),
    (["__iter__"], unknown),
    (["__le__"], [(other,)]),
    (["__len__"], None),
    (["__lshift__"], [(other,)]),
    (["__lt__"], [(other,)]),
    (["__matmul__"], [(other,)]),
    (["__mul__"], [(other,)]),
    (["__ne__"], [(other,)]),
    (["__neg__"], [(other,)]),
    (["__pow__"], [(other,)]),
    (["__radd__"], [(other,)]),
    # (["__repr__"], None),
    # (["__rfloordiv__"], [(other,)]),
    # (["__rlshift__", "__rlshift__", None], unknown),
    # (["__rmatmul__", "__rmatmul__", None], unknown),
    # (["__rmul__"], [(other,)]),
    # (["__rpow__"], [(other,)]),
    # (["__rrshift__", "__rrshift__", None], unknown),
    # (["__rshift__"], unknown),
    # (["__rsub__"], [(other,)]),
    # (["__rtruediv__"], unknown),
    # (["__sizeof__"], unknown),
    # (["__str__"], unknown),
    # (["__sub__"], [(other,)]),
    # (["__truediv__"], unknown),
    # (["argmax"], [(0,)]),
    # (["argmin"], [(0,)]),
    # (["argsort"], [(-1,)]),
    # (["choose"], unknown),
    # (["clip"], [(-2,)]),
    # (["copy", "copy", "clone"], None),
    # (["cumprod"], [(0,)]),
    # (["cumsum"], [(0,)]),
    # (["diagonal"], [(0, )]),
    # (["dot"], unknown),
    # (["flat"], unknown),
    # (["flatten"], None),
    # (["item"], unknown),
    # (["itemset"], unknown),
    # (["itemsize"], unknown),
    # (["max"], None),
    # (["mean"], None),
    # (["min"], None),
    # (["ndim"], None),
    # (["prod"], [(other,)]),
    # (["repeat"], unknown),
    # (["reshape"], [(-1,)]),
    # (["resize"], [(-1, )]),
    # (["sort"], unknown),
    # (["squeeze"], unknown),
    # (["std"], unknown),
    # (["sum"], unknown),
    # (["swapaxes"], unknown),
    # (["T"], None),
    # (["take"], unknown),
    # (["transpose"], unknown),
]

def convert_other(input_structure, other_tensor):
    def replace_other(input):
        return other_tensor if input is other else input

    new_input_structure = []
    for input in input_structure:
        if isinstance(input, tuple):
            new_input_structure.append(tuple(replace_other(elem) for elem in input))
        elif isinstance(input, dict):
            new_input_structure.append({k: replace_other(input) for k, v in input.items()})
        else:
            raise ValueError("Not the right parameter structure")

    return new_input_structure

@pytest.mark.parametrize("fcall_list, args", test_methods)
def test_ops(fcall_list, args, tensor_pairs):
    if args is unknown:
        raise ValueError("Please specify a list of args for the function!")

    if args is None:
        no_of_arg_tests = 1
    else:
        no_of_arg_tests = len(args)

    results = []
    for i, (a, b) in enumerate(tensor_pairs):
        if len(fcall_list) == 1:
            fcall = fcall_list[0]
        elif len(fcall_list) == 3:
            fcall = fcall_list[i]
        else:
            raise ValueError("Bad format when specifying the fcall list")

        if fcall is None:
            results.append([None for _ in range(no_of_arg_tests)])
            continue

        if args is None:
            method = getattr(type(a), fcall)

            if callable(method):
                result = method(a)
            elif isgetsetdescriptor(method) or isdatadescriptor(method):
                result = getattr(a, fcall)
            else:
                raise ValueError("Please provide the right type of arguments for the fcall")
            results.append([result])
        else:
            tensor_specific_args = convert_other(args, b)
            sub_results = []
            for args_scenario in tensor_specific_args:
                method = getattr(a, fcall)
                if isinstance(args_scenario, tuple):
                    result = method(*args_scenario)
                elif isinstance(args_scenario, dict):
                    result = method(**args_scenario)
                else:
                    raise ValueError("Args not in the right structure")
                sub_results.append(result)

            results.append(sub_results)

    SYFT_RESULT = 0
    NUMPY_RESULT = 1
    TORCH_RESULT = 2

    for i in range(no_of_arg_tests):
        syft_result = results[SYFT_RESULT][i]
        numpy_result = results[NUMPY_RESULT][i]
        torch_result = results[TORCH_RESULT][i]

        while hasattr(syft_result, "child"):
            syft_result = syft_result.child


        assert syft_result is not None

        if numpy_result is not None:
            assert np.array_equal(syft_result, numpy_result)

        if torch_result is not None:
            assert np.array_equal(syft_result, torch_result.numpy())