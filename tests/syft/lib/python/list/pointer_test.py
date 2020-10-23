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

inputs = {
    "__add__": [[[1, 2, 3]], [3], [[1, 2.5, True]]],
    "__contains__": [[41], [-1], [None], ["test"]],
    "__eq__": [[[41, 15, 3, 80]], [[True]]],
    "__ge__": [[[41, 15, 3, 80]], [[True]]],
    "__getitem__": [[1], [2], [3]],
    "__gt__": [[[41, 15, 3, 80]], [[True]]],
    "__iadd__": [[[1, 2, 3]], [3], [[1, 2.5, True]]],
    "__imul__": [[[1, 2, 3]], [3], [[1, 2.5, True]]],
    "__le__": [[[41, 15, 3, 80]], [[True]]],
    "__len__": [[]],
    "__lt__": [[[41, 15, 3, 80]], [[True]]],
    "__mul__": [[[1, 2, 3]], [3], [[1, 2.5, True]]],
    "__ne__": [[[41, 15, 3, 80]], [[True]]],
    "__reversed__": [[]],
    "__rmul__": [[[1, 2, 3]], [3], [[1, 2.5, True]]],
    "__setitem__": [[0, 2], [1, 5]],
    "__sizeof__": [[]],
    "append": [[1], [2], [3]],
    "clear": [[]],
    "copy": [[]],
    "count": [[]],
    "extend": [[[1, 2, 3]], [[4, 5, 6]]],
    "index": [[0], [1], [5]],
    "insert": [[0, "a"], [3, "b"]],
    "pop": [[0], [3]],
    "remove": [[1], [42]],
    "reverse": [[]],
    "sort": [[], [True]],
}

objects = [
    ([41, 15, 3, 80], sy.lib.python.List([41, 15, 3, 80]), remote_python.List([41, 15, 3, 80])),
    (list(range(2**8)), sy.lib.python.List(list(range(2**8))), remote_python.List(list(range(2**8))))
]

@pytest.mark.parametrize("test_objects", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(test_objects, func):
    py_obj, sy_obj, remote_sy_obj = test_objects

    py_method = getattr(py_obj, func)
    sy_method = getattr(sy_obj, func)
    remote_sy_method = getattr(remote_sy_obj, func)

    possible_inputs = inputs[func]

    for possible_input in possible_inputs:
        py_res, py_e, sy_res, sy_e, remote_sy = None, None, None, None, None

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

        get_permission(remote_sy_obj)
        assert py_obj == sy_obj
        assert sy_obj == remote_sy_obj.get()

@pytest.mark.parametrize("test_objects", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_iterator(test_objects, func):
    pass
