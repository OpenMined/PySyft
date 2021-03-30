# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy


def get_permission(obj: Any, node: sy.VirtualMachine) -> None:
    remote_obj = node.store[obj.id_at_location]
    remote_obj.read_permissions[node.verify_key] = obj.id_at_location


inputs = {
    "__add__": [["str_1"], [""]],
    "__contains__": [[""], ["tu"], ["AND"], ["\t"]],
    "__eq__:": [["george"], ["test"]],
    "__ge__": [["test"], ["aaaatest"]],
    "__getitem__": [[0], [2]],
    "__gt__": [["test"], ["aaaatest"]],
    "__le__": [["test"], ["aaaatest"]],
    "__len__": [[]],
    "__lt__": [["test"], ["aaaatest"]],
    "__mul__": [[2], [3]],
    "__ne__": [["george"], ["test"]],
    "__radd__": [["str_1"], [""]],
    "__rmul__": [[2], [3]],
    "capitalize": [[]],
    "casefold": [[]],
    "center": [[15], [10, "0"]],
    "count": [["a"], ["g"], ["g", 5, 10]],
    "endswith": [["or"], ["ge"]],
    "expandtabs": [[]],
    "find": [["test"], ["tudor"], ["george"]],
    "index": [["test"]],
    "isalnum": [[]],
    "isalpha": [[]],
    "isascii": [[]],
    "isdecimal": [[]],
    "isdigit": [[]],
    "isidentifier": [[]],
    "islower": [[]],
    "isnumeric": [[]],
    "isprintable": [[]],
    "isspace": [[]],
    "istitle": [[]],
    "isupper": [[]],
    "join": [[["test", "me"]], [("tst", "m")]],
    "ljust": [[10], [25]],
    "lower": [[]],
    "lstrip": [[]],
    "partition": [["a"], ["t"], ["\t"]],
    "replace": [["tud", "dut"], ["AN", "KEY"]],
    "rfind": [["test"], ["tudor"], ["george"]],
    "rindex": [["test"]],
    "rjust": [[10], [25]],
    "rpartition": [["a"], ["t"], ["\t"]],
    "rsplit": [[], ["a"]],
    "rstrip": [["e"], ["r"], []],
    "split": [[], ["a"]],
    "splitlines": [[]],
    "startswith": [["j"], ["O"]],
    "strip": [["t"], ["r"], []],
    "swapcase": [[]],
    "title": [[]],
    "upper": [[]],
    "zfill": [[5], [10]],
}

objects = ["Op en Min ed", "george", "J A Y", "ANDREW", "tud\t\nor"]


@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("func", inputs.keys())
def test_pointer_objectives(
    test_object, func, node: sy.VirtualMachine, client: sy.VirtualMachineClient
):
    remote_python = client.syft.lib.python.String
    local_constructor = sy.lib.python.String

    py_obj, sy_obj, remote_sy_obj = (
        test_object,
        local_constructor(test_object),
        remote_python(test_object),
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
            get_permission(remote_sy_res, node)
            remote_sy_res = remote_sy_res.get()
        except Exception as remote_sy_e:
            remote_sy_res = str(remote_sy_e)

        assert py_res == sy_res
        assert sy_res == remote_sy_res
