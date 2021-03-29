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
    "__add__": [[[1, 2, 3]], [[1, 2.5, True]]],
    "__contains__": [[41], [-1], [None], ["test"]],
    "__eq__": [[[41, 15, 3, 80]], [[True]]],
    "__ge__": [[[41, 15, 3, 80]], [[True]]],
    "__getitem__": [[1], [2], [3]],
    "__gt__": [[[41, 15, 3, 80]], [[True]]],
    "__iadd__": [[[1, 2, 3]], [[1, 2.5, True]]],
    "__imul__": [[1], [3], [5]],
    "__le__": [[[41, 15, 3, 80]], [[True]]],
    "__len__": [[]],
    "__lt__": [[[41, 15, 3, 80]], [[True]]],
    "__mul__": [[1], [3], [5]],
    "__ne__": [[[41, 15, 3, 80]], [[True]]],
    "__rmul__": [[1], [3], [5]],
    "__setitem__": [[0, 2], [1, 5]],
    "append": [[1], [2], [3]],
    "clear": [[]],
    "copy": [[]],
    "count": [[1], [2]],
    "extend": [[[1, 2, 3]], [[4, 5, 6]]],
    "index": [[0], [1], [5]],
    "insert": [[0, "a"], [3, "b"]],
    "pop": [[0], [3]],
    "remove": [[1], [42]],
    "reverse": [[]],
    "sort": [[]],
}


def generate_0():
    return ([41, 15, 3, 80],)


def generate_1():
    return (list(range(2 ** 5)),)


mapping = {0: generate_0, 1: generate_1}

objects = [0, 1]


@pytest.mark.xfail
@pytest.mark.slow
@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(
    test_object, func, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    py_obj = mapping[test_object]()
    sy_obj, remote_sy_obj = sy.lib.python.List(py_obj), client.syft.lib.python.List(
        py_obj
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
        #
        # if isinstance(py_res, float):
        #     py_res = int(py_res * 1000) / 1000
        #     sy_res = int(sy_res * 1000) / 1000
        #     remote_sy_res = int(remote_sy_res * 1000) / 1000

        assert py_res == sy_res
        assert sy_res == remote_sy_res

        assert py_obj == sy_obj
        # TODO add this as well when the store logic will work
        get_permission(remote_sy_obj, node, client)
        assert sy_obj == remote_sy_obj.get()


@pytest.mark.slow
@pytest.mark.parametrize("test_object", objects)
def test_iterator(test_object, client: sy.VirtualMachineClient):
    py_obj = mapping[test_object]()
    sy_obj, remote_sy_obj = sy.lib.python.List(py_obj), client.syft.lib.python.List(
        py_obj
    )

    py_iter = iter(py_obj)
    sy_iter = iter(sy_obj)

    remote_sy_obj.set_request_config({})
    rsy_iter = iter(remote_sy_obj)

    for i in range(len(py_obj)):
        py_elem = next(py_iter)
        sy_elem = next(sy_iter)
        rsy_elem = next(rsy_iter)

        assert py_elem == sy_elem
        assert sy_elem == rsy_elem.get()


@pytest.mark.slow
@pytest.mark.parametrize("test_object", objects)
def test_reversed_iterator(test_object, client):
    py_obj = mapping[test_object]()
    sy_obj, remote_sy_obj = sy.lib.python.List(py_obj), client.syft.lib.python.List(
        py_obj
    )

    py_iter = reversed(py_obj)
    sy_iter = reversed(sy_obj)
    rsy_iter = reversed(remote_sy_obj)

    for i in range(len(py_obj)):
        py_elem = next(py_iter)
        sy_elem = next(sy_iter)
        rsy_elem = next(rsy_iter)

        assert py_elem == sy_elem
        assert sy_elem == rsy_elem.get()
