# syft absolute
import syft as sy
from syft.lib.python.int import Int
from syft.proto.lib.python.int_pb2 import Int as Int_PB


def test_serde() -> None:
    syft_int = Int(5)

    serialized = syft_int._object2proto()

    assert isinstance(serialized, Int_PB)

    deserialized = Int._proto2object(proto=serialized)

    assert isinstance(deserialized, Int)
    assert deserialized == syft_int


def test_send(client: sy.VirtualMachineClient) -> None:
    syft_int = Int(5)
    ptr = syft_int.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "IntPointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_int


def test_int_bytes() -> None:
    # Testing if multiple serialization of the similar object results in same bytes
    syft_string_1 = Int(7)
    syft_string_2 = Int(7)
    assert sy.serialize(syft_string_1, to_bytes=True) == sy.serialize(
        syft_string_2, to_bytes=True
    )
