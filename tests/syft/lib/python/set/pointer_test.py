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
    "__and__": [],
    "__eq__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__ge__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__gt__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__iand__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__ior__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__isub__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__ixor__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__le__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__len__": [[]],
    "__lt__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__ne__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__or__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__sub__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "__xor__": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "add": [[10], [24], [42], [-1]],
    "clear": [[]],
    "difference": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "difference_update": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "discard": [[10], [24], [42], [-1]],
    "intersection": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "intersection_update": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "isdisjoint": [{24}],
    "issuperset": [{24}],
    "pop": [[]],
    "remove": [[10], [24], [42], [-1]],
    "symmetric_difference_update": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "symmetric_difference": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "union": [[{42, 24}], [set(range(10))], [set(range(25))]],
    "update": [[{42, 24}], [set(range(10))], [set(range(25))]],
}

objects = [[24], [42], [24, 42], list(range(10))]


@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(
    test_object, func, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    py_obj, sy_obj, remote_sy_obj = (
        set(test_object),
        sy.lib.python.Set(test_object),
        client.syft.lib.python.Set(test_object),
    )
    possible_inputs = inputs[func]

    if not hasattr(py_obj, func):
        return

    py_method = getattr(py_obj, func)
    sy_method = getattr(sy_obj, func)

    if func == "__len__":
        func = "len"

    remote_sy_method = getattr(remote_sy_obj, func)

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
