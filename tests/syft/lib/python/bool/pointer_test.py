# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy

sy.VERBOSE = False
alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_root_client()
remote_python = alice_client.syft.lib.python


def get_permission(obj: Any) -> None:
    remote_obj = alice.store[obj.id_at_location]
    remote_obj.read_permissions[alice_client.verify_key] = obj.id_at_location


sy_true = sy.lib.python.Bool(True)
sy_false = sy.lib.python.Bool(False)

sy_true_ptr = remote_python.Bool(True)
sy_false_ptr = remote_python.Bool(False)

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
objects = [(True, sy_true, sy_true_ptr), (False, sy_false, sy_false_ptr)]


@pytest.mark.parametrize("test_objects", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(test_objects, func):
    py_obj, sy_obj, remote_sy_obj = test_objects

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
            get_permission(remote_sy_res)
            remote_sy_res = remote_sy_res.get()
        except Exception as remote_sy_e:
            remote_sy_res = str(remote_sy_e)

        if isinstance(py_res, float):
            py_res = int(py_res * 1000) / 1000
            sy_res = int(sy_res * 1000) / 1000
            remote_sy_res = int(remote_sy_res * 1000) / 1000

        assert py_res == sy_res
        assert sy_res == remote_sy_res


@pytest.mark.parametrize(
    "test_objects", [(True, sy_true, sy_true_ptr), (False, sy_false, sy_false_ptr)]
)
@pytest.mark.parametrize("property", properties)
def test_pointer_properties(test_objects, property):
    py_obj, sy_obj, remote_sy_obj = test_objects

    # TODO add support for proper properties

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
        get_permission(remote_sy_res)
        remote_sy_res = remote_sy_res.get()
    except Exception as remote_sy_e:
        remote_sy_res = str(remote_sy_e)

    if isinstance(py_res, float):
        py_res = int(py_res * 1000) / 1000
        sy_res = int(sy_res * 1000) / 1000
        remote_sy_res = int(remote_sy_res * 1000) / 1000

    assert py_res == sy_res
    assert sy_res == remote_sy_res
