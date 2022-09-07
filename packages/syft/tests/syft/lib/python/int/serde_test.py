# syft absolute
import syft as sy
from syft.lib.python.int import Int


def test_serde() -> None:
    syft_int = Int(5)

    serialized = sy.serialize(syft_int)

    deserialized = sy.deserialize(serialized)

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
