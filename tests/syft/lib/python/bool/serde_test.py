# syft absolute
import syft as sy
from syft.lib.python.bool import Bool
from syft.proto.lib.python.bool_pb2 import Bool as Bool_PB


def test_serde() -> None:
    syft_bool = Bool(True)

    serialized = syft_bool._object2proto()

    assert isinstance(serialized, Bool_PB)

    deserialized = Bool._proto2object(proto=serialized)

    assert isinstance(deserialized, Bool)
    assert deserialized.id == syft_bool.id
    assert deserialized == syft_bool


def test_send(client: sy.VirtualMachineClient) -> None:
    syft_bool = Bool(5)
    ptr = syft_bool.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "BoolPointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_bool
