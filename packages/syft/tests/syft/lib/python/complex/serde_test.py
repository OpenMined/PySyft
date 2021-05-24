# syft absolute
import syft as sy
from syft.lib.python.complex import Complex
from syft.proto.lib.python.complex_pb2 import Complex as Complex_PB


def test_serde() -> None:
    syft_complex = Complex("2+3j")

    serialized = syft_complex._object2proto()

    assert isinstance(serialized, Complex_PB)

    deserialized = Complex._proto2object(proto=serialized)

    assert isinstance(deserialized, Complex)
    assert deserialized.id == syft_complex.id
    assert deserialized == syft_complex


def test_send(client: sy.VirtualMachineClient) -> None:
    syft_complex = Complex("2+3j")
    ptr = syft_complex.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "ComplexPointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res == syft_complex
