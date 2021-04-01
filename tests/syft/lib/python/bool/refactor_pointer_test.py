# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy
from tests.syft.lib.python.refactored.refactor_pointer import pointer_objectives, pointer_properties


inputs = {
    "__abs__": [[]],
    "__add__": [[True], [False], [42]],
    "__and__": [[True], [False]],
    "__ceil__": [[]],
    "__divmod__": [[42], [256]],
    "__eq__": [[True], [False]],
    "__float__": [[]],
    "__floor__": [[]],
    "__floordiv__": [[42], [256]],
    "__ge__": [[True], [False]],
    "__gt__": [[True], [False]],
    "__invert__": [[]],
    "__le__": [[True], [False], [42]],
    "__lshift__": [[42]],
    "__lt__": [[True], [False], [42]],
    "__mod__": [[42], [256]],
    "__mul__": [[42], [256]],
    "__ne__": [[True], [False], [42], [256]],
    "__neg__": [[]],
    "__or__": [[True], [False]],
    "__pos__": [[]],
    "__pow__": [[42], [256]],
    "__radd__": [[42], [256]],
    "__rand__": [[True], [False]],
    "__rdivmod__": [[42], [256]],
    "__rfloordiv__": [[42], [256]],
    "__rlshift__": [[42], [256]],
    "__rmod__": [[42], [256]],
    "__rmul__": [[42], [256]],
    "__ror__": [[True], [False]],
    "__round__": [[]],
    "__rpow__": [[42], [256]],
    "__rrshift__": [[42], [256]],
    "__rshift__": [[42], [256]],
    "__rsub__": [[42], [256]],
    "__rtruediv__": [[42], [256]],
    "__rxor__": [[True], [False]],
    "__sub__": [[42], [256]],
    "__truediv__": [[42], [256]],
    "__xor__": [[True], [False]],
    "__trunc__": [[]],
    "as_integer_ratio": [[]],
    "bit_length": [[]],
    "conjugate": [[]],
}

properties = ["denominator", "numerator", "imag", "real"]
objects = [True, False]


@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(
    test_object, func, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    py_obj, sy_obj, remote_sy_obj = (
        test_object,
        sy.lib.python.Bool(test_object),
        client.syft.lib.python.Bool(test_object),
    )

    test_obj = py_obj, sy_obj, remote_sy_obj

    possible_inputs = inputs[func]

    if not hasattr(py_obj, func):
        return

    for possible_input in possible_inputs:
        pointer_objectives(test_obj, func, node,
                           client, possible_input, "BOOL")


@pytest.mark.parametrize("test_object", [True, False])
@pytest.mark.parametrize("property", properties)
def test_pointer_properties(
    test_object, property, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    py_obj, sy_obj, remote_sy_obj = (
        test_object,
        sy.lib.python.Bool(test_object),
        client.syft.lib.python.Bool(test_object),
    )

    # TODO add support for proper properties

    test_obj = py_obj, sy_obj, remote_sy_obj
    pointer_properties(test_obj, property, node, client, "BOOL")
