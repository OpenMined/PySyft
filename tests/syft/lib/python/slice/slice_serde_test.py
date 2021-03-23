# syft absolute
import syft as sy
from syft.lib.python.slice import Slice
from syft.proto.lib.python.slice_pb2 import Slice as Slice_PB


def test_slice_serde() -> None:
    syft_slice = Slice(1, 3)

    serialized = syft_slice._object2proto()

    assert isinstance(serialized, Slice_PB)

    deserialized = Slice._proto2object(proto=serialized)

    assert isinstance(deserialized, Slice)
    assert deserialized.id == syft_slice.id
    assert deserialized == syft_slice


def test_slice_send() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()


    syft_slice = Slice(1, 3)
    ptr = syft_slice.send(alice_client)

    # Check pointer type
    assert ptr.__class__.__name__ == "SlicePointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_slice
