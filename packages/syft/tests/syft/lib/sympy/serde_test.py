# third party
import pytest

# syft absolute
import syft as sy
from syft import deserialize
from syft import serialize


@pytest.mark.vendor(lib="sympy")
def test_variable_serde() -> None:
    # third party
    from sympy.core.symbol import Symbol

    sy.load("sympy")

    x = Symbol("x")

    protobuf_obj = serialize(x)
    deserialized = deserialize(protobuf_obj)

    assert x == deserialized
    assert x.name == deserialized.name


# @pytest.mark.vendor(lib="sympy")
# def test_multi_child_serde() -> None:
#     # third party
#     from sympy.core.symbol import Symbol
#     from sympy.core.add import Add
#     from sympy.core.mul import Mul

#     sy.load("sympy")

#     x = Symbol("x")

#     multi_child_types = [
#         (x + x, Sum),
#         (x * x, Product),
#     ]

#     for result, result_type in multi_child_types:
#         assert isinstance(result, result_type)

#         protobuf_obj = serialize(result)
#         deserialized = deserialize(protobuf_obj)

#         assert result == deserialized

#     assert True is False
