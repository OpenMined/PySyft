# stdlib
from collections import OrderedDict
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy

alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_root_client()
remote_python = alice_client.syft.lib.python


def get_permission(obj: Any) -> None:
    remote_obj = alice.store[obj.id_at_location]
    remote_obj.read_permissions[alice_client.verify_key] = obj.id_at_location


inputs = {
    "__contains__": [["1"], ["2"], [1], [2]],
    "__delitem__": [["1"], [1], [10], ["10"], [500], [{1: 1}]],
    "__eq__": [
        [OrderedDict([("1", 1), ("2", 2), ("3", 3)])],
        [OrderedDict([("1", 1), ("2", 2), ("3", 3), ("4", 4)])],
    ],
    "__getitem__": [["1"], [1], [10], ["10"], [500], [{1: 1}]],
    "__len__": [[]],
    "__ne__": [
        [OrderedDict([("1", 1), ("2", 2), ("3", 3)])],
        [OrderedDict([("1", 1), ("2", 2), ("3", 3), ("4", 4)])],
    ],
    "__setitem__": [["1", None], [1, None], [10, None], ["10", 10], [500, None]],
    "clear": [[]],
    "copy": [[]],
    "fromkeys": [[[1, 2, 3]], [["1", 2, "4"]]],
    "get": [[1], [2], [4], [-1]],
    "items": [[]],
    "keys": [[]],
    "move_to_end": [[1], [2], [50], [-1]],
    "pop": [[1], [2], [50], [-1]],
    "popitem": [[True], [False]],
    "setdefault": [[1, "1"], [-1, 1], ["1", 1]],
    "update": [[{1: 1}], [{1: 1, 2: 2, -1: 5}]],
    "values": [[]],
}

objects = [
    (
        OrderedDict([("1", 1), ("2", 2), ("3", 3)]),
        sy.lib.python.collections.OrderedDict([("1", 1), ("2", 2), ("3", 3)]),
        remote_python.collections.OrderedDict([("1", 1), ("2", 2), ("3", 3)]),
    ),
    (
        OrderedDict([(i, j) for i, j in zip(range(100), range(100))]),
        sy.lib.python.collections.OrderedDict(
            [(i, j) for i, j in zip(range(100), range(100))]
        ),
        remote_python.collections.OrderedDict(
            [(i, j) for i, j in zip(range(100), range(100))]
        ),
    ),
]


@pytest.mark.parametrize("test_objects", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(test_objects, func):
    py_obj, sy_obj, remote_sy_obj = test_objects
    possible_inputs = inputs[func]

    if not hasattr(py_obj, func):
        return

    py_method = getattr(py_obj, func)

    if func == "get":
        func = "dict_get"

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
            get_permission(remote_sy_res)
            remote_sy_res = remote_sy_res.get()
        except Exception as remote_sy_e:
            remote_sy_res = str(remote_sy_e)

        if isinstance(py_res, float):
            py_res = int(py_res * 1000) / 1000
            sy_res = int(sy_res * 1000) / 1000
            remote_sy_res = int(remote_sy_res * 1000) / 1000

        if func in ["items", "values", "keys"]:
            py_res = list(py_res)

        assert py_res == sy_res
        assert sy_res == remote_sy_res
