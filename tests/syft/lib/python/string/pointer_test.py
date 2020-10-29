# stdlib
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

objects = [
    (
        "Op en Min ed",
        sy.lib.python.String("Op en Min ed"),
        remote_python.String("Op en Min ed"),
    ),
    ("george", sy.lib.python.String("george"), remote_python.String("george")),
    ("J A Y", sy.lib.python.String("J A Y"), remote_python.String("J A Y")),
    ("ANDREW", sy.lib.python.String("ANDREW"), remote_python.String("ANDREW")),
    ("tud\t\nor", sy.lib.python.String("tud\t\nor"), remote_python.String("tud\t\nor")),
]


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

        assert py_res == sy_res
        assert sy_res == remote_sy_res
