# stdlib
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

# third party
import pytest

# syft absolute
import syft as sy


def get_permission(
    obj: Any, node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    remote_obj = node.store[obj.id_at_location]
    remote_obj.read_permissions[client.verify_key] = obj.id_at_location


inputs_float: Dict[str, List] = {
    "__abs__": [[]],
    "__add__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__divmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__eq__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__floordiv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__ge__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__gt__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__le__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__lt__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__mod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__mul__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__ne__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__neg__": [[]],
    "__pos__": [[]],
    "__pow__": [[0], [1], [2]],
    "__rdivmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rfloordiv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rmod__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rmul__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__round__": [[]],
    "__rpow__": [[0], [1]],
    "__rsub__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__rtruediv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__sub__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__truediv__": [[0], [42], [2 ** 10], [-(2 ** 10)]],
    "__trunc__": [[]],
    "as_integer_ratio": [[]],
    "conjugate": [[]],
}

inputs_string: Dict[str, List] = {
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

inputs_bool: Dict[str, List] = {
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

inputs_set: Dict[str, List] = {
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

inputs_ordered_dict: Dict[str, List] = {
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

inputs_int: Dict[str, List] = {
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

inputs_list: Dict[str, List] = {
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
inputs_dict: Dict[str, List] = {
    "__contains__": [["a"], ["d"]],
    "__eq__": [[{"a": 1, "b": 2, "c": None}], [{1: "a", 2: "b"}]],
    "__getitem__": [["a"], [1]],
    "__hash__": [[]],
    "__len__": [[]],
    "__ne__": [[{"a": 1, "b": 2, "c": None}], [{1: "a", 2: "b"}]],
    "__str__": [[]],
    "copy": [[]],
    "fromkeys": [[[("a", 1), ("b", 2), ("c", 2)]]],
    "items": [[]],
    "keys": [[]],
    "values": [[]],
    "pop": [["a"]],
    "popitem": [[]],
    "setdefault": [["start", 101]],
    "clear": [[]],
    "get": [["a"]],
}

test_dict: Dict[str, Dict[str, Any]] = {
    "float": {
        "inputs": inputs_float,
        "construct": (
            float,
            sy.lib.python.Float,
            lambda client: client.syft.lib.python.Float,
        ),
        "objects": [0.5, 42.5, 2 ** 10 + 0.5, -(2 ** 10 + 0.5)],
    },
    "int": {
        "inputs": inputs_int,
        "construct": (
            int,
            sy.lib.python.Int,
            lambda client: client.syft.lib.python.Int,
        ),
        "objects": [0, 42, 2 ** 10, -(2 ** 10)],
        "properties": ["denominator", "numerator", "imag", "real"],
    },
    "bool": {
        "inputs": inputs_bool,
        "construct": (
            bool,
            sy.lib.python.Bool,
            lambda client: client.syft.lib.python.Bool,
        ),
        "objects": [True, False],
        "properties": ["denominator", "numerator", "imag", "real"],
    },
    "set": {
        "inputs": inputs_set,
        "construct": (
            set,
            sy.lib.python.Set,
            lambda client: client.syft.lib.python.Set,
        ),
        "objects": [[24], [42], [24, 42], list(range(10))],
    },
    "string": {
        "inputs": inputs_string,
        "construct": (
            str,
            sy.lib.python.String,
            lambda client: client.syft.lib.python.String,
        ),
        "objects": ["Op en Min ed", "george", "J A Y", "ANDREW", "tud\t\nor"],
    },
    "ordered_dict": {
        "inputs": inputs_ordered_dict,
        "construct": (
            OrderedDict,
            sy.lib.python.collections.OrderedDict,
            lambda client: client.syft.lib.python.collections.OrderedDict,
        ),
        "objects": [
            [("1", 1), ("2", 2), ("3", 3)],
            list(zip(range(100), range(100))),
        ],
    },
    "list": {
        "inputs": inputs_list,
        "construct": (
            list,
            sy.lib.python.List,
            lambda client: client.syft.lib.python.List,
        ),
        "objects": [lambda: ([41, 15, 3, 80],), lambda: (list(range(2 ** 5)),)],
    },
    "dict": {
        "inputs": inputs_dict,
        "construct": (
            dict,
            sy.lib.python.Dict,
            lambda client: client.syft.lib.python.Dict,
        ),
        "objects": [[("a", 1), ("b", 2), ("c", None)], {1: "a", 2: "b"}],
    },
}


parameters_pointer_objectives = []
for py_type in test_dict:
    for test_object in test_dict[py_type]["objects"]:
        for func in test_dict[py_type]["inputs"]:
            if py_type == "list":
                parameters_pointer_objectives.append([py_type, test_object(), func])
                # test_object in list are lambda func
            else:
                parameters_pointer_objectives.append([py_type, test_object, func])


@pytest.mark.slow
@pytest.mark.parametrize("py_type,test_object,func", parameters_pointer_objectives)
def test_pointer_objectives(
    py_type: str,
    test_object: Any,
    func: str,
    node: sy.VirtualMachine,
    client: sy.VirtualMachineClient,
) -> None:
    py_construct, sy_construct, remote_sy_construct_fn = test_dict[py_type]["construct"]
    remote_sy_construct = remote_sy_construct_fn(client)

    py_obj, sy_obj, remote_sy_obj = (
        py_construct(test_object),
        sy_construct(test_object),
        remote_sy_construct(test_object),
    )

    if not hasattr(py_obj, func):
        return

    possible_inputs = test_dict[py_type]["inputs"][func]

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
            get_permission(remote_sy_res, node, client)
            remote_sy_res = remote_sy_res.get()
        except Exception as remote_sy_e:
            remote_sy_res = str(remote_sy_e)

        if isinstance(py_res, float):
            py_res = int(py_res * 1000) / 1000
            sy_res = int(sy_res * 1000) / 1000
            remote_sy_res = int(remote_sy_res * 1000) / 1000

        if func in ["items", "values", "keys", "popitem"]:
            py_res = list(py_res)
            sy_res = list(sy_res)

        assert py_res == sy_res
        # TODO: support `.get` for IteratorPointer objects
        if func not in ("items", "keys", "values", "popitem"):
            assert sy_res == remote_sy_res


parameters_pointer_properties = []
for py_type in test_dict:
    if "properties" in test_dict[py_type]:
        parameters_pointer_properties += [
            [py_type, test_object, property]
            for test_object in test_dict[py_type]["objects"]
            for property in test_dict[py_type]["properties"]
        ]


@pytest.mark.parametrize("py_type,test_object,property", parameters_pointer_properties)
def test_pointer_properties(
    py_type: str,
    test_object: Any,
    property: str,
    node: sy.VirtualMachine,
    client: sy.VirtualMachineClient,
) -> None:
    py_construct, sy_construct, remote_sy_construct_fn = test_dict[py_type]["construct"]
    remote_sy_construct = remote_sy_construct_fn(client)
    py_obj, sy_obj, remote_sy_obj = (
        py_construct(test_object),
        sy_construct(test_object),
        remote_sy_construct(test_object),
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


@pytest.mark.slow
@pytest.mark.parametrize("test_object_fn", test_dict["list"]["objects"])
# Test iter method on list and dict object.
def test_list_iterator(
    test_object_fn: Callable[[], List[int]], client: sy.VirtualMachineClient
) -> None:
    py_obj = test_object_fn()
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
@pytest.mark.parametrize("test_object", test_dict["dict"]["objects"])
def test_dict_iterator(test_object: List, client: sy.VirtualMachineClient) -> None:
    py_obj = test_object
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
@pytest.mark.parametrize("test_object_fn", test_dict["list"]["objects"])
def test_reversed_iterator(
    test_object_fn: Callable[[], List[int]], client: sy.VirtualMachineClient
) -> None:
    py_obj = test_object_fn()
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
