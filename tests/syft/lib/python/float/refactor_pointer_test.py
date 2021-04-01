from typing import Any

# third party
import pytest

# syft absolute
import syft as sy
from tests.syft.lib.python.refactored.refactor_pointer import pointer_objectives


inputs = {
    "__abs__": [[]],
    "__add__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__divmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__eq__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__floordiv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__ge__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__gt__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__le__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__lt__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__mod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__mul__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__ne__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__neg__": [[]],
    "__pos__": [[]],
    "__pow__": [[0], [1], [2]],
    "__rdivmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rfloordiv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rmul__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__round__": [[]],
    "__rpow__": [[0], [1]],
    "__rsub__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rtruediv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__sub__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__truediv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__trunc__": [[]],
    "as_integer_ratio": [[]],
    "conjugate": [[]],
}

properties = ["denominator", "numerator", "imag", "real"]

objects = [0.5, 42.5, 2 ** 10 + 0.5, -(2 ** 10 + 0.5)]


@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(
    test_object, func, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    py_obj, sy_obj, remote_sy_obj = (
        test_object,
        sy.lib.python.Float(test_object),
        client.syft.lib.python.Float(test_object),
    )

    test_obj = py_obj, sy_obj, remote_sy_obj

    possible_inputs = inputs[func]

    if not hasattr(py_obj, func):
        return

    for possible_input in possible_inputs:
        pointer_objectives(test_obj, func, node,
                           client, possible_input, "FLOAT")
