# syft absolute
import syft as sy
from syft.lib.python.bytes import Bytes


def test_bytes_bytes() -> None:
    # Testing if multiple serialization of the similar object results in same bytes
    value_1 = Bytes(b"hello")
    value_2 = Bytes(b"hello")
    assert sy.serialize(value_1, to_bytes=True) == sy.serialize(value_2, to_bytes=True)
