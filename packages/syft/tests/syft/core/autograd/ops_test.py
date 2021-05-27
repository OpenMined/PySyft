# stdlib
from inspect import isdatadescriptor
from inspect import isgetsetdescriptor

# third party
import numpy as np
import pytest
import torch as th
import operator

# syft absolute
from syft.core.tensor.tensor import Tensor

other = type("Other", tuple(), {})
unknown = type("Unknown", tuple(), {})

constant_numpy_3_3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

def generate_numpy_pair():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return (a, np.array(np.flip(a)))


def generate_torch_pair():
    numpy_tensor_pair = generate_numpy_pair()
    a = th.tensor(numpy_tensor_pair[0], requires_grad=True)
    b = th.tensor(numpy_tensor_pair[1], requires_grad=True)
    return (a, b)


def generate_syft_pair():
    numpy_tensor_pair = generate_numpy_pair()
    return (
        Tensor(numpy_tensor_pair[0]).autograd(requires_grad=True),
        Tensor(numpy_tensor_pair[1]).autograd(requires_grad=True),
    )


@pytest.fixture(scope="function")
def tensor_pairs():
    return (generate_syft_pair(), generate_numpy_pair(), generate_torch_pair())


# Example on how to add tests:
# (["fcall_name_if_it's_uniform"], None <- it's a fcall with no arguments or a property)
# (["fcall_syft", "fcall_numpy", "fcall_torch"], [(0, ), {"data": None}]) <- a list with multiple
# input scenarios, namely:
# 1. you can pass a tuple specifying the *args
# 2. you can pass a dict specifying the **kwargs

test_methods = [
    (["__mul__"], [(other,), (2,), (constant_numpy_3_3,)]),
    (["__rmul__"], [(other,), (2,), (constant_numpy_3_3,)]),
    ([operator.mul, "rhs"], [(other,), (2,), (constant_numpy_3_3,)]),
    ([operator.mul, "lhs"], [(other,), (2,), (constant_numpy_3_3,)])
    # (["__abs__"], None),
    # (["__add__"], [(other,)]),
    # (["__divmod__", "__divmod__", None], [(other,)]),  # no backward
    # (["__eq__"], [(other,)]),  # no backward
    # (["__floordiv__"], [(other,)]),  # no backward
    # (["__ge__"], [(other,)]),  # no backward
    # (["__getitem__"], [(0,), (range(0, 2),)]),
    # (["__gt__"], [(other,)]),  # no backward
    # (["__index__"], unknown), # delete_later
    # (["__invert__"], None),   # no backward
    # # (["__iter__"], unknown),  # use getitem
    # (["__le__"], [(other,)]),  # no backward
    # (["__len__"], None),  # no backward
    # (["__lshift__"], [(3,)]),  # no backward
    # (["__lt__"], [(other,)]),  # no backward
    # (["__matmul__"], [(other,)]),
    # (["__ne__"], [(other,)]),  # no backward
    # (["__neg__"], None),
    # (["__pow__"], [(other,)]),  # broken
    # (["__radd__"], [(other,)]),
    # # (["__repr__"], None),  # no backward
    # (["__rfloordiv__"], [(other,)]),  # no backward
    # (["__rlshift__", "__rlshift__", None], unknown),  # no backward
    # (["__rmatmul__", "__rmatmul__", "__matmul__"], [(other,)]),
    # (["__rmul__"], [(other,), (2, )]),
    # Tudor: fix multiple test arguments (reset gradients)
    # (["__rpow__"], [(2,)]),  # add (other,)  # broken
    # (["__rrshift__", "__rrshift__", None], unknown),  # no backward
    # # (["__rshift__"], unknown),  # no backward
    # (["__rsub__"], [(other,)]),
    # # (["__rtruediv__"], unknown),
    # # (["__sizeof__"], unknown),  # no backward
    # # (["__str__"], unknown),  # no backward
    # (["__sub__"], [(other,)]),
    # # (["__truediv__"], unknown),
    # (["argmax"], [(0,)]),  # no backward
    # (["argmin"], [(0,)]),  # no backward
    # (["argsort"], [(-1,)]),  # no backward
    # # (["choose"], unknown), # no backward
    # (["clip"], [(-2,)]),  # no backward
    # (["copy", "copy", "clone"], None),
    # (["cumprod"], [(0,)]),
    # (["cumsum"], [(0,)]),
    # (["diagonal"], [(0,)]),
    # (["dot"], unknown),  # no backward
    # (["flat"], unknown),  # no backward
    # (["flatten"], None),  # no backward
    # (["item"], unknown),  # no backward
    # (["itemset"], unknown),  # no backward
    # (["itemsize"], unknown),  # no backward
    # (["max"], None),
    # (["mean"], None),
    # (["min"], None),
    # (["ndim"], None),  # no backward
    # (["prod"], None),
    # # (["repeat"], unknown),
    # (["reshape"], [(-1,)]),
    # # (["resize"], [(-1, )]),
    # # (["sort"], unknown),
    # # (["squeeze"], unknown),
    # # (["std"], unknown),
    # (["sum"], unknown),
    # # (["swapaxes"], unknown),  # no backward
    # (["T"], None), # no backward
    # # (["take"], unknown),
    # # (["transpose"], unknown),
]


def generate_test_methods(test_methods):
    new_test_methods = []
    for (fname, args) in test_methods:
        if args is not None:
            for arg in args:
                if type(fname[0]) is str:
                    new_test_methods.append((fname[0], fname, arg))
                else:
                    new_test_methods.append((fname[0].__name__ + "_" + fname[1], fname, arg))
        else:
            new_test_methods.append((fname[0] + "_None", fname, None))

    return new_test_methods

__test_methods = generate_test_methods(test_methods=test_methods)


def convert_other(input_structure, other_tensor):
    def replace_other(input):
        return other_tensor if input is other else input

    if isinstance(input_structure, tuple):
        return tuple(replace_other(elem) for elem in input_structure)
    elif isinstance(input_structure, dict):
        return {k: replace_other(v) for k, v in input_structure.items()}
    else:
        raise ValueError("Not the right parameter structure")

# handle converting collections of th.Tensor to numpy
def th_as_np(t):
    if hasattr(t, "requires_grad"):
        t = t.detach()

    if isinstance(t, (int, bool, float)):
        return t
    if isinstance(t, th.Tensor):
        return t.numpy()
    if isinstance(t, tuple):
        return tuple([x.numpy() for x in t])
    if isinstance(t, set):
        return set([x.numpy() for x in t])
    if isinstance(t, list):
        return list([x.numpy() for x in t])
    if isinstance(t, np.ndarray):
        return t

    raise Exception(f"unknown type {type(t)} {t}")


@pytest.mark.parametrize("_test_name, fcall_list, args", __test_methods)
def test_forward_pass(_test_name, fcall_list, args, tensor_pairs):
    if args is unknown:
        raise ValueError("Please specify a list of args for the function!")

    results = []
    for i, (a, b) in enumerate(tensor_pairs):
        if len(fcall_list) == 1:
            fcall = fcall_list[0]
        elif len(fcall_list) == 3:
            fcall = fcall_list[i]
        else:
            fcall = fcall_list[0]

        if fcall is None:
            results.append(None)
            continue
        if type(fcall) is str:
            # testing method
            if args is None:
                method = getattr(type(a), fcall)

                if callable(method):
                    result = method(a)
                elif isgetsetdescriptor(method) or isdatadescriptor(method):
                    result = getattr(a, fcall)
                else:
                    raise ValueError(
                        "Please provide the right type of arguments for the fcall"
                    )
                results.append(result)
            else:

                tensor_specific_args = convert_other(args, b)
                method = getattr(a, fcall)
                if isinstance(tensor_specific_args, tuple):
                    result = method(*tensor_specific_args)
                elif isinstance(tensor_specific_args, dict):
                    result = method(**tensor_specific_args)
                else:
                    raise ValueError("Args not in the right structure")

                results.append(result)
        else:
            # testing operator
            operator = fcall_list[0]
            position = fcall_list[1]
            tensor_specific_args = convert_other(args, b)
            if len(tensor_specific_args) == 1:
                if type(a) is th.Tensor and type(tensor_specific_args[0]) is np.ndarray:
                    a = a.detach().numpy()

            if position is "lhs":
                result = operator(a, tensor_specific_args[0])
            elif position is "rhs":
                result = operator(tensor_specific_args[0], a)
            elif position is "single":
                result = operator(a)
            else:
                raise ValueError("Wrong pos")

            results.append(result)

    SYFT_RESULT = 0
    NUMPY_RESULT = 1
    TORCH_RESULT = 2

    syft_result = results[SYFT_RESULT]
    numpy_result = results[NUMPY_RESULT]
    torch_result = results[TORCH_RESULT]

    syft_original_result = syft_result
    while hasattr(syft_result, "child"):
        syft_result = syft_result.child

    assert syft_result is not None

    if numpy_result is not None:
        assert np.array_equal(syft_result, numpy_result)

    if torch_result is not None and torch_result is not NotImplemented:
        assert np.array_equal(syft_result, th_as_np(torch_result))
    else:
        return

    # backward pass tests
    if not hasattr(syft_original_result, "requires_grad") or not hasattr(torch_result,
                                                                         "requires_grad"):
        return

    assert syft_original_result.requires_grad == torch_result.requires_grad

    if not syft_original_result.requires_grad:
        return

    syft_original_result.backward()
    th.sum(torch_result).backward()
    syft_tensor_pairs = tensor_pairs[SYFT_RESULT]
    torch_tensor_pairs = tensor_pairs[TORCH_RESULT]

    for i in range(len(syft_tensor_pairs)):
        syft_tensor_grad = syft_tensor_pairs[i].grad
        torch_tensor_grad = torch_tensor_pairs[i].grad
        if torch_tensor_grad is None:
            continue
        torch_tensor_grad = th_as_np(torch_tensor_grad)
        assert np.array_equal(syft_tensor_grad.data_child, torch_tensor_grad)
