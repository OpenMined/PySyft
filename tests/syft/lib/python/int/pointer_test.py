# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy


def get_permission(
    obj: Any, node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    remote_obj = node.store[obj.id_at_location]
    remote_obj.read_permissions[client.verify_key] = obj.id_at_location


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

    if not hasattr(py_obj, func):
        return

    py_method = getattr(py_obj, func)
    sy_method = getattr(sy_obj, func)
    remote_sy_method = getattr(remote_sy_obj, func)

    possible_inputs = inputs[func]

    for possible_input in possible_inputs:
        try:
            py_res = py_method(*possible_input)
        except Exception as py_e:
            py_res = str(py_e)

        try:
            sy_res = sy_method(*possible_input)
        except Exception as sy_e:
            sy_res = str(sy_e)

        try:
            remote_sy_res = remote_sy_method(*possible_input)
            get_permission(remote_sy_res, node, client)
            remote_sy_res = remote_sy_res.get()
        except Exception as remote_sy_e:
            remote_sy_res = str(remote_sy_e)

        if isinstance(py_res, float):
            py_res = int(py_res * 1000) / 1000
            sy_res = int(sy_res * 1000) / 1000
            remote_sy_res = int(remote_sy_res * 1000) / 1000

        assert py_res == sy_res
        assert sy_res == remote_sy_res


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

    try:
        py_res = getattr(py_obj, property)
    except Exception as py_e:
        py_res = str(py_e)

    try:
        sy_res = getattr(sy_obj, property)()
    except Exception as sy_e:
        sy_res = str(sy_e)

    try:
        remote_sy_res = getattr(remote_sy_obj, property)()
        get_permission(remote_sy_res, node, client)
        remote_sy_res = remote_sy_res.get()
    except Exception as remote_sy_e:
        remote_sy_res = str(remote_sy_e)

    if isinstance(py_res, float):
        py_res = int(py_res * 1000) / 1000
        sy_res = int(sy_res * 1000) / 1000
        remote_sy_res = int(remote_sy_res * 1000) / 1000

    assert py_res == sy_res
    assert sy_res == remote_sy_res
