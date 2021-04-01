# syft absolute
import syft as sy
from syft.lib.python.set import Set
from syft.proto.lib.python.set_pb2 import Set as Set_PB


def test_serde() -> None:
    syft_int = Set([1, 2, 3, 4])

    serialized = syft_int._object2proto()

    assert isinstance(serialized, Set_PB)

    deserialized = Set._proto2object(proto=serialized)

    assert isinstance(deserialized, Set)
    assert deserialized.id == syft_int.id
    assert deserialized == syft_int


def test_send(client: sy.VirtualMachineClient) -> None:
    syft_int = Set([1, 2, 3, 4])
    ptr = syft_int.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "SetPointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_int
