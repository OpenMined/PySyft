# third party
import pytest

# syft absolute
import syft as sy
from syft import deserialize
from syft import serialize


@pytest.mark.vendor(lib="pymbolic")
def test_variable_serde() -> None:
    # third party
    from pymbolic.primitives import Variable

    sy.load("pymbolic")

    x = Variable("x")

    protobuf_obj = serialize(x)
    deserialized = deserialize(protobuf_obj)

    assert x == deserialized


@pytest.mark.vendor(lib="pymbolic")
def test_product_serde() -> None:
    # third party
    from pymbolic.primitives import Product
    from pymbolic.primitives import Variable

    sy.load("pymbolic")

    x = Variable("x")
    y = x * x

    protobuf_obj = serialize(y)
    deserialized = deserialize(protobuf_obj)

    assert y == deserialized
    assert Product(children=(x, x)) == deserialized


@pytest.mark.vendor(lib="pymbolic")
def test_sum_serde() -> None:
    # third party
    from pymbolic.primitives import Sum
    from pymbolic.primitives import Variable

    sy.load("pymbolic")

    x = Variable("x")
    y = x + x

    protobuf_obj = serialize(y)
    deserialized = deserialize(protobuf_obj)

    assert y == deserialized
    assert Sum(children=(x, x)) == deserialized
