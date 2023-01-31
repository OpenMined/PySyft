# syft absolute
import syft as sy
from syft.lib.python.string import String


def test_string_serde() -> None:
    syft_string = String("Hello OpenMined")

    serialized = sy.serialize(syft_string)

    deserialized = sy.deserialize(serialized)

    assert isinstance(deserialized, String)


def test_string_send(client: sy.VirtualMachineClient) -> None:
    syft_string = String("Hello OpenMined!")
    ptr = syft_string.send(client)

    # Check pointer type
    assert ptr.__class__.__name__ == "StringPointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_string


def test_string_bytes() -> None:
    # Testing if multiple serialization of the similar object results in same bytes
    syft_string_1 = String("Hello OpenMined")
    syft_string_2 = String("Hello OpenMined")
    assert sy.serialize(syft_string_1, to_bytes=True) == sy.serialize(
        syft_string_2, to_bytes=True
    )
