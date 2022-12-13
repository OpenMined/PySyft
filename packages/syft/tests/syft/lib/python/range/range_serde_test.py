# syft absolute
import syft as sy
from syft.lib.python.range import Range
from syft.proto.lib.python.range_pb2 import Range as Range_PB


def test_range_serde() -> None:
    syft_range = Range(5, 1, -1)

    serialized = syft_range._object2proto()

    assert isinstance(serialized, Range_PB)

    deserialized = Range._proto2object(proto=serialized)

    assert isinstance(deserialized, Range)
    for deserialized_el, original_el in zip(deserialized, syft_range):
        assert deserialized_el == original_el


def test_range_send(client: sy.VirtualMachineClient) -> None:
    syft_range = Range(5)
    ptr = syft_range.send(client)

    assert ptr.__class__.__name__ == "RangePointer"

    res = ptr.get()
    for res_el, original_el in zip(res, syft_range):
        assert res_el == original_el


def test_range_bytes() -> None:
    # Testing if multiple serialization of the similar object results in same bytes
    value_1 = Range(7)
    value_2 = Range(7)
    assert sy.serialize(value_1, to_bytes=True) == sy.serialize(value_2, to_bytes=True)
