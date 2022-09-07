# syft absolute
import syft as sy
from syft.lib.python.set import Set


def test_serde() -> None:
    syft_int = Set([1, 2, 3, 4])

    serialized = sy.serialize(syft_int)

    deserialized = sy.deserialize(serialized)

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
