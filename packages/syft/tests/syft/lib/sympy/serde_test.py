# third party
import pytest

# syft absolute
import syft as sy
from syft import deserialize
from syft import serialize


@pytest.mark.vendor(lib="sympy")
def test_symbol_serde() -> None:
    # third party
    from sympy.core.symbol import Symbol

    sy.load("sympy")

    x = Symbol("x")

    protobuf_obj = serialize(x)
    deserialized = deserialize(protobuf_obj)

    assert x == deserialized
    assert x.name == deserialized.name


@pytest.mark.vendor(lib="sympy")
def test_add_serde() -> None:
    # third party
    from sympy.core.add import Add
    from sympy.core.symbol import Symbol

    sy.load("sympy")

    x = Symbol("x")
    y = Symbol("y")
    z = x + y

    protobuf_obj = serialize(z)
    deserialized = deserialize(protobuf_obj)

    assert z == deserialized
    assert z._args == (x, y)
    assert Add(x, y) == z


@pytest.mark.vendor(lib="sympy")
def test_mul_serde() -> None:
    # third party
    from sympy.core.mul import Mul
    from sympy.core.symbol import Symbol

    sy.load("sympy")

    x = Symbol("x")
    y = Symbol("y")
    z = x * y

    protobuf_obj = serialize(z)
    deserialized = deserialize(protobuf_obj)

    assert z == deserialized
    assert z._args == (x, y)
    assert Mul(x, y) == z


@pytest.mark.vendor(lib="sympy")
def test_integer_serde() -> None:
    # third party
    from sympy.core.numbers import Integer

    sy.load("sympy")

    python_x = 2
    x = Integer(python_x)

    protobuf_obj = serialize(x)
    deserialized = deserialize(protobuf_obj)

    assert x == deserialized
    assert x.p == python_x
    assert Integer(python_x) == x


@pytest.mark.vendor(lib="sympy")
def test_rational_serde() -> None:
    # third party
    from sympy.core.numbers import Rational

    sy.load("sympy")

    python_x = 1 / 3
    x = Rational(1, q=3)

    protobuf_obj = serialize(x)
    print("type", protobuf_obj, type(protobuf_obj))
    deserialized = deserialize(protobuf_obj)

    assert x == deserialized
    assert (x.p / x.q) == python_x
    assert Rational(1, q=3) == x
