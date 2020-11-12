# syft absolute
import syft as sy
from syft.lib.python.float import Float
from syft.lib.python.float import FloatWrapper
from syft.proto.lib.python.float_pb2 import Float as Float_PB


def test_serde() -> None:
    syft_float = Float(5)

    serialized = syft_float._object2proto()

    assert isinstance(serialized, Float_PB)

    deserialized = Float._proto2object(proto=serialized)

    assert isinstance(deserialized, Float)
    assert deserialized.id == syft_float.id
    assert deserialized == syft_float


def test_send() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    syft_float = Float(5)
    ptr = syft_float.send(alice_client)
    # Check pointer type
    assert ptr.__class__.__name__ == "FloatPointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_float


def test_protobuf():
    assert Float.get_protobuf_schema()
    assert FloatWrapper.get_wrapped_type() is Float
