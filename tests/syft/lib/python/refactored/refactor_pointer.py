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


# @pytest.mark.parametrize("test_object", objects)
# @pytest.mark.parametrize("func", inputs.keys())
def pointer_objectives(
    obj: Any, func, node: sy.VirtualMachine, client: sy.VirtualMachineClient, input, type: str
):
    py_obj, sy_obj, remote_sy_obj = obj
    possible_input = input

    if not hasattr(py_obj, func):
        return

    py_method = getattr(py_obj, func)
    sy_method = None
    if type == "OD":
        if func == "get":
            func = "dict_get"

        sy_method = getattr(sy_obj, func)

        if func == "__len__":
            func = "len"
    if not sy_method:
        sy_method = getattr(sy_obj, func)
    remote_sy_method = getattr(remote_sy_obj, func)

    # for possible_input in possible_inputs:
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

# if type == "OD":
    if func in ["items", "values", "keys"]:
        py_res = list(py_res)
        sy_res = list(sy_res)

    assert py_res == sy_res

    if func not in ("items", "keys", "values"):
        assert sy_res == remote_sy_res


def pointer_properties(
    obj: Any, property, node: sy.VirtualMachine, client: sy.VirtualMachineClient, type: str
):
    py_obj, sy_obj, remote_sy_obj = obj
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
