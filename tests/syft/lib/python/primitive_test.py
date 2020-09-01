# stdlib
import json
import math
import sys
import uuid

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.lib.python.primitive import PyPrimitive
from syft.lib.python.primitive import isprimitive


def get_uid() -> UID:
    return UID(value=uuid.UUID(int=333779996850170035686993356951732753684))


def get_none_bytes() -> bytes:
    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}}
    container = {
        "objType": "syft.lib.python.primitive.PyPrimitive",
        "content": json.dumps(content),
    }
    return bytes(json.dumps(container), "utf-8")


def get_bool_true_bytes() -> bytes:
    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}, "type": "BOOL", "int": "1"}
    container = {
        "objType": "syft.lib.python.primitive.PyPrimitive",
        "content": json.dumps(content),
    }
    return bytes(json.dumps(container), "utf-8")


def get_bool_false_bytes() -> bytes:
    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}, "type": "BOOL"}
    container = {
        "objType": "syft.lib.python.primitive.PyPrimitive",
        "content": json.dumps(content),
    }
    return bytes(json.dumps(container), "utf-8")


def get_int_bytes() -> bytes:
    content = {
        "id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="},
        "type": "INT",
        "int": "9223372036854775807",
    }
    container = {
        "objType": "syft.lib.python.primitive.PyPrimitive",
        "content": json.dumps(content),
    }
    return bytes(json.dumps(container), "utf-8")


def get_pi_bytes() -> bytes:
    content = {
        "id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="},
        "type": "FLOAT",
        "float": 3.141592653589793,
    }
    container = {
        "objType": "syft.lib.python.primitive.PyPrimitive",
        "content": json.dumps(content),
    }
    return bytes(json.dumps(container), "utf-8")


def get_nan_bytes() -> bytes:
    content = {
        "id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="},
        "type": "FLOAT",
        "float": "NaN",
    }
    container = {
        "objType": "syft.lib.python.primitive.PyPrimitive",
        "content": json.dumps(content),
    }
    return bytes(json.dumps(container), "utf-8")


def get_ne_inf_bytes() -> bytes:
    content = {
        "id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="},
        "type": "FLOAT",
        "float": "-Infinity",
    }
    container = {
        "objType": "syft.lib.python.primitive.PyPrimitive",
        "content": json.dumps(content),
    }
    return bytes(json.dumps(container), "utf-8")


def get_str_bytes() -> bytes:
    content = {
        "id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="},
        "type": "STRING",
        "str": "Hello Proto 🙂",
    }
    container = {
        "objType": "syft.lib.python.primitive.PyPrimitive",
        "content": json.dumps(content),
    }
    return bytes(json.dumps(container), "utf-8")


# if type(value) in [type(None), bool, int, float, str]:
def test_isprimitive() -> None:
    """Checks all the basic types and collections except set are working"""

    a = None
    b = True
    c = False
    d = 1
    e = 1.1
    f = "string"

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

    assert type(f) is str
    assert isprimitive(value=f)


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


def test_none_type() -> None:
    """Tests the None type behaves as expected"""
    a = None
    pyprim_a = PyPrimitive(data=a)
    b = False

    assert a != b
    assert isinstance(a, type(None))
    assert a == pyprim_a.data
    assert a == pyprim_a
    assert b != pyprim_a


def test_bool_type() -> None:
    """Tests the bool type behaves as expected"""
    a = True
    b = False
    pyprim_a = PyPrimitive(data=a)
    pyprim_a2 = PyPrimitive(data=a)
    pyprim_b = PyPrimitive(data=b)
    pyprim_int = PyPrimitive(data=1)
    a_int = 1

    assert a != b
    assert pyprim_a != pyprim_b
    assert pyprim_a == pyprim_a2
    assert pyprim_a is pyprim_a
    assert pyprim_a is not pyprim_a2
    assert isinstance(a, bool)
    assert pyprim_a.__bool__() is True
    assert pyprim_b.__bool__() is False
    assert a == a_int  # normal python
    assert a is not a_int  # normal python
    assert pyprim_a == pyprim_int  # same behavior
    assert pyprim_a is not pyprim_int  # same behavior
    assert a == pyprim_a.data
    assert a == pyprim_a
    assert b == pyprim_b.data
    assert b == pyprim_b
    assert pyprim_a > pyprim_b
    assert pyprim_b < pyprim_a
    assert pyprim_a >= pyprim_b
    assert pyprim_b <= pyprim_a
    assert not (pyprim_b >= pyprim_a)
    assert not (pyprim_a <= pyprim_b)


def test_int_type() -> None:
    """Tests the int type behaves as expected"""
    a = 1
    b = 0
    c = -1
    pyprim_a = PyPrimitive(data=a)
    pyprim_a2 = PyPrimitive(data=a)
    pyprim_b = PyPrimitive(data=b)
    pyprim_c = PyPrimitive(data=c)
    a_bool = True
    b_bool = False

    assert a != b
    assert pyprim_a != pyprim_b
    assert pyprim_a != pyprim_c
    assert pyprim_a == pyprim_a2
    assert pyprim_a == a
    assert pyprim_b == b
    assert pyprim_c == c
    assert pyprim_a is not pyprim_a2
    assert c < b and b < a
    assert (pyprim_c < pyprim_b) and (pyprim_b < pyprim_a)
    assert a > b and b > c
    assert pyprim_a > pyprim_b and pyprim_b > pyprim_c
    assert pyprim_a >= pyprim_b
    assert pyprim_b >= pyprim_b
    assert pyprim_a == a_bool
    assert pyprim_a is not a_bool
    assert pyprim_b == b_bool
    assert pyprim_b is not b_bool
    assert a + c == b
    assert pyprim_a + pyprim_c == pyprim_b
    assert 2 * a == 2
    assert 2 * pyprim_a == (pyprim_a + pyprim_a)
    assert pyprim_b == (pyprim_a - pyprim_a)
    assert pyprim_a / pyprim_a == pyprim_a


def test_float_type() -> None:
    """Tests the float type behaves as expected"""
    a = 1.0
    b = 0.0
    c = -1.0
    pyprim_a = PyPrimitive(data=a)
    pyprim_a2 = PyPrimitive(data=a)
    pyprim_b = PyPrimitive(data=b)
    pyprim_c = PyPrimitive(data=c)
    a_int = 1
    b_int = 0

    assert a != b
    assert pyprim_a != pyprim_b
    assert pyprim_a != pyprim_c
    assert pyprim_a == pyprim_a2
    assert pyprim_a == a
    assert pyprim_b == b
    assert pyprim_c == c
    assert pyprim_a is not pyprim_a2
    assert c < b and b < a
    assert (pyprim_c < pyprim_b) and (pyprim_b < pyprim_a)
    assert a > b and b > c
    assert pyprim_a > pyprim_b and pyprim_b > pyprim_c
    assert pyprim_a >= pyprim_b
    assert pyprim_b >= pyprim_b
    assert pyprim_a == a_int
    assert pyprim_a is not a_int
    assert pyprim_b == b_int
    assert pyprim_b is not b_int
    assert a + c == b
    assert pyprim_a + pyprim_c == pyprim_b
    assert 2 * a == 2.0
    assert 2 * pyprim_a == (pyprim_a + pyprim_a)
    assert pyprim_b == (pyprim_a - pyprim_a)
    assert pyprim_a / pyprim_a == pyprim_a


def test_str_type() -> None:
    """Tests the str type behaves as expected"""
    a = "a"
    b = "b"
    c = "c"
    pyprim_a = PyPrimitive(data=a)
    pyprim_a2 = PyPrimitive(data=a)
    pyprim_b = PyPrimitive(data=b)
    pyprim_c = PyPrimitive(data=c)
    a_int = 1
    b_int = 0

    assert a != b
    assert pyprim_a != pyprim_b
    assert pyprim_a != pyprim_c
    assert pyprim_a == pyprim_a2
    assert pyprim_a == a
    assert pyprim_a is not a
    assert pyprim_b == b
    assert pyprim_b is not b
    assert pyprim_c == c
    assert pyprim_a is not pyprim_a2
    assert c > b and b > a
    assert (pyprim_c > pyprim_b) and (pyprim_b > pyprim_a)
    assert a < b and b < c
    assert pyprim_a < pyprim_b and pyprim_b < pyprim_c
    assert pyprim_b >= pyprim_a
    assert pyprim_a <= pyprim_a
    assert pyprim_a != a_int
    assert pyprim_a is not a_int
    assert pyprim_b != b_int
    assert pyprim_b is not b_int
    assert a + c == "ac"
    assert pyprim_a + pyprim_c == "ac"
    assert 2 * a == "aa"
    assert 2 * pyprim_a == (pyprim_a + pyprim_a)
    assert len(a) == 1
    assert len(a * 2) == 2
    print(type(len(a * 2)), len(a * 2))


def test_serde_primitive_none() -> None:
    """Serialize / Deserialize primitives"""

    a = PyPrimitive(data=None, id=get_uid())
    a_se = a.serialize(to_binary=True)

    assert a_se == get_none_bytes()

    de = sy.deserialize(blob=a_se, from_json=True, from_binary=True)

    assert de == a
    assert de.data is None
    assert de.id == get_uid()


def test_serde_primitive_bool() -> None:
    """Serialize / Deserialize primitives"""

    a = PyPrimitive(data=True, id=get_uid())
    a_se = a.serialize(to_binary=True)

    b = PyPrimitive(data=False, id=get_uid())
    b_se = b.serialize(to_binary=True)

    assert a_se == get_bool_true_bytes()
    assert b_se == get_bool_false_bytes()

    de_true = sy.deserialize(blob=a_se, from_json=True, from_binary=True)
    de_false = sy.deserialize(blob=b_se, from_json=True, from_binary=True)

    assert de_true == a
    assert de_true.data is True
    assert de_true.id == get_uid()

    assert de_false == b
    assert de_false.data is False
    assert de_false.id == get_uid()


def test_serde_primitive_int() -> None:
    """Serialize / Deserialize primitives"""

    a = PyPrimitive(data=sys.maxsize, id=get_uid())
    a_se = a.serialize(to_binary=True)

    assert a_se == get_int_bytes()

    de = sy.deserialize(blob=a_se, from_json=True, from_binary=True)

    assert sys.maxsize == 9223372036854775807
    assert de == 9223372036854775807
    assert de.data == 9223372036854775807
    assert de.id == get_uid()


def test_serde_primitive_float() -> None:
    """Serialize / Deserialize primitives"""

    a = PyPrimitive(data=float("-inf"), id=get_uid())
    a_se = a.serialize(to_binary=True)

    b = PyPrimitive(data=float("nan"), id=get_uid())
    b_se = b.serialize(to_binary=True)

    c = PyPrimitive(data=math.pi, id=get_uid())
    c_se = c.serialize(to_binary=True)

    assert a_se == get_ne_inf_bytes()
    assert b_se == get_nan_bytes()
    assert c_se == get_pi_bytes()

    de_a = sy.deserialize(blob=a_se, from_json=True, from_binary=True)
    de_b = sy.deserialize(blob=b_se, from_json=True, from_binary=True)
    de_c = sy.deserialize(blob=c_se, from_json=True, from_binary=True)

    assert de_a == float("-inf")
    assert de_a.data == float("-inf")
    assert de_a.id == get_uid()

    assert de_b != de_b  # because NaN doesnt equal itself
    assert math.isnan(de_b.data)
    assert de_b.id == get_uid()

    assert de_c == 3.141592653589793
    assert de_c.data == math.pi
    assert de_c.id == get_uid()


def test_serde_primitive_str() -> None:
    """Serialize / Deserialize primitives"""

    a = PyPrimitive(data="Hello Proto 🙂", id=get_uid())
    a_se = a.serialize(to_binary=True)

    assert a_se == get_str_bytes()

    de_a = sy.deserialize(blob=a_se, from_json=True, from_binary=True)

    assert de_a == "Hello Proto 🙂"
    assert len(de_a) == 13
    assert de_a.data == "Hello Proto 🙂"
    assert len(de_a.data) == 13
    assert de_a.id == get_uid()
