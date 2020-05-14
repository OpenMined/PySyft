import pytest
import syft

from syft.serde import msgpack
from syft.serde import protobuf

from syft.serde.msgpack.native_serde import MAP_NATIVE_SIMPLIFIERS_AND_DETAILERS

msgpack.serde.init_global_vars_msgpack()
protobuf.serde.init_global_vars()


@pytest.mark.parametrize(
    "cls", msgpack.serde.simplifiers.keys() - dict(MAP_NATIVE_SIMPLIFIERS_AND_DETAILERS).keys()
)
def test_msgpack_parity(cls):
    """Checks all types in msgpck serde are also covered by Protobuf"""
    assert (
        cls in dict(protobuf.serde.get_bufferizers()).keys()
    ), f"{cls} doesn't have a Protobuf schema."
