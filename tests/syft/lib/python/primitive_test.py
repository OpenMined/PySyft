import json
from typing import Any
from typing import Dict

from syft.lib.python.primitive import isprimitive

# from syft.lib.python.primitive import PythonPrimitive
# import syft as sy


def get_py_dict() -> Dict[str, Any]:
    python_dict = {
        "a": [None, True, False, 1, 1.1],
        "b": [
            [None, True, False, 1, 1.1],
            [None, True, False, 1, 1.1],
            [None, True, False, 1, 1.1],
            [None, True, False, 1, 1.1],
        ],
    }

    return python_dict


def get_proto_bytes() -> bytes:
    content = {
        "id": {"value": "9r3ZeY/4RiKUwSEN2A35Ow=="},
        "objType": "dict",
        "content": json.dumps(get_py_dict()),
    }
    envelope = {
        "objType": "syft.lib.python.primitive.PythonPrimitive",
        "content": json.dumps(content),
    }
    blob = bytes(json.dumps(envelope), "utf-8")
    return blob


# if type(value) in [type(None), bool, int, float]:
def test_isprimitive() -> None:
    """Checks all the basic types and collections except set are working"""

    a = None
    b = True
    c = False
    d = 1
    e = 1.1

    assert isinstance(a, type(None))
    assert isprimitive(value=a)

    assert type(b) is bool
    assert isprimitive(value=b)

    assert type(c) is bool
    assert isprimitive(value=c)

    assert type(d) is int
    assert isprimitive(value=d)

    assert type(e) is float
    assert isprimitive(value=e)


# if type(value) in [tuple, list]:
# if type(value) is dict:
def test_isprimitive_collections() -> None:
    """Checks all the basic types and collections except set are working"""

    a = None
    b = True
    c = False
    d = 1
    e = 1.1

    bottom_list = [a, b, c, d, e]
    bottom_tuple = (a, b, c, d, e)
    container_list = [bottom_list, bottom_tuple, bottom_list, bottom_tuple]
    container_dict = {"a": bottom_tuple, "b": container_list}

    assert type(bottom_list) is list
    assert isprimitive(value=bottom_list)

    assert type(bottom_tuple) is tuple
    assert isprimitive(value=bottom_tuple)

    assert type(container_list) is list
    assert isprimitive(value=container_list)

    assert type(container_dict) is dict
    assert isprimitive(value=container_dict)


def test_not_isprimitive() -> None:
    """Checks for types that fail"""

    def a() -> None:
        pass

    b = set([1, 2, 3])
    c = type(None)
    d = list
    e = bytes(1)
    bad_keys = {None: "test"}  # JSON only allows str keys

    assert type(a).__name__ == "function"
    assert not isprimitive(value=a)

    assert type(b) is set
    assert not isprimitive(value=b)

    assert c.__name__ == "NoneType"  # type: ignore
    assert not isprimitive(value=c)

    assert type(d) is type
    assert not isprimitive(value=d)

    assert type(e) is bytes
    assert not isprimitive(value=e)

    assert type(bad_keys) is dict
    assert not isprimitive(value=bad_keys)


# TODO: Fix
# def test_serialize_primitives() -> None:
#     """Serialize primitives to proto / binary"""

#     # copy ID for comparison
#     de = sy.deserialize(blob=get_proto_bytes(), from_json=True, from_binary=True)

#     obj = PythonPrimitive(id=de.id, value=get_py_dict())
#     ser = obj.serialize(to_binary=True)
#     assert ser == get_proto_bytes()


# TODO: Fix
# def test_deserialize_primitives() -> None:
#     """Deserialize primitives back from proto / binary"""

#     de = sy.deserialize(blob=get_proto_bytes(), from_json=True, from_binary=True)

#     assert type(de.value) == dict
#     assert de.value == get_py_dict()
