from collections import OrderedDict
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy
from tests.syft.lib.python.refactored.refactor_pointer import pointer_objectives


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
    [("1", 1), ("2", 2), ("3", 3)],
    list(zip(range(100), range(100))),
]


@pytest.mark.slow
@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(
    test_object, func, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    py_obj, sy_obj, remote_sy_obj = (
        OrderedDict(test_object),
        sy.lib.python.collections.OrderedDict(test_object),
        client.syft.lib.python.collections.OrderedDict(test_object),
    )

    test_obj = py_obj, sy_obj, remote_sy_obj

    possible_inputs = inputs[func]

    if not hasattr(py_obj, func):
        return

    for possible_input in possible_inputs:
        pointer_objectives(test_obj, func, node,
                           client, possible_input, "OD")
