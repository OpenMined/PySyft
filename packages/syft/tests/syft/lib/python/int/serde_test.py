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
    assert deserialized.id == syft_int.id
    assert deserialized == syft_int


def test_send(client: sy.VirtualMachineClient) -> None:
    syft_int = Int(5)
    ptr = syft_int.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "IntPointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_int
