# stdlib
import sys

# syft absolute
import syft as sy
from syft.lib.python.float import Float
from syft.proto.lib.python.float_pb2 import Float as Float_PB


def test_serde() -> None:
    syft_float = Float(5)

    serialized = syft_float._object2proto()

    assert isinstance(serialized, Float_PB)

    deserialized = Float._proto2object(proto=serialized)

    assert isinstance(deserialized, Float)
    assert deserialized.id == syft_float.id
    assert deserialized == syft_float


def test_send(client: sy.VirtualMachineClient) -> None:
    syft_float = Float(5)
    ptr = syft_float.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "FloatPointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_float


def test_python_double() -> None:
    max_float = sys.float_info.max
    ser = sy.serialize(max_float, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert max_float == de

    min_float = sys.float_info.min
    ser = sy.serialize(min_float, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert min_float == de

    neg_max_float = -sys.float_info.max
    ser = sy.serialize(neg_max_float, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert neg_max_float == de

    neg_min_float = -sys.float_info.min
    ser = sy.serialize(neg_min_float, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert neg_min_float == de
