# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy
from tests.syft.lib.python.refactored.refactor_pointer import pointer_objectives, pointer_properties


inputs = {
    "__abs__": [[]],
    "__add__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__and__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__ceil__": [[]],
    "__divmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__eq__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__float__": [[]],
    "__floor__": [[]],
    "__floordiv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__ge__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__gt__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__invert__": [[]],
    "__le__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__lshift__": [[42]],
    "__lt__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__mod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__mul__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__ne__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__neg__": [[]],
    "__or__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__pos__": [[]],
    "__pow__": [[0], [1], [2]],
    "__radd__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rand__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rdivmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rfloordiv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rlshift__": [[0]],
    "__rmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rmul__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__ror__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__round__": [[]],
    "__rpow__": [[0], [1]],
    "__rrshift__": [[0], [42]],
    "__rshift__": [[0], [42]],
    "__rsub__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rtruediv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rxor__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__sub__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__truediv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__xor__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__trunc__": [[]],
    "as_integer_ratio": [[]],
    "bit_length": [[]],
    "conjugate": [[]],
}

properties = ["denominator", "numerator", "imag", "real"]

objects = [0, 42, 2 ** 10, -(2 ** 10)]


@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(
    test_object, func, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    py_obj, sy_obj, remote_sy_obj = (
        test_object,
        sy.lib.python.Int(test_object),
        client.syft.lib.python.Int(test_object),
    )

    test_obj = py_obj, sy_obj, remote_sy_obj

    possible_inputs = inputs[func]

    if not hasattr(py_obj, func):
        return

    for possible_input in possible_inputs:
        pointer_objectives(test_obj, func, node,
                           client, possible_input, "INT")


@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("property", properties)
def test_pointer_properties(
    test_object, property, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    py_obj, sy_obj, remote_sy_obj = (
        test_object,
        sy.lib.python.Int(test_object),
        client.syft.lib.python.Int(test_object),
    )

    test_obj = py_obj, sy_obj, remote_sy_obj
    pointer_properties(test_obj, property, node, client, "INT")
