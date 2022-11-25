# syft absolute
import syft as sy
from syft.lib.python.string import String
from syft.proto.lib.python.string_pb2 import String as String_PB


def test_string_serde() -> None:
    syft_string = String("Hello OpenMined")

    serialized = syft_string._object2proto()

    assert isinstance(serialized, String_PB)

    deserialized = String._proto2object(proto=serialized)

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
    # Testing if multiple serialization of the same object results in same bytes
    syft_string = String("Hello OpenMined")
    assert sy.serialize(syft_string, to_bytes=True) == sy.serialize(
        syft_string, to_bytes=True
    )
