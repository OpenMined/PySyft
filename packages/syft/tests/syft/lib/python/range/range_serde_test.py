# syft absolute
import syft as sy
from syft.lib.python.range import Range


def test_range_serde() -> None:
    syft_range = Range(5, 1, -1)

    serialized = sy.serialize(syft_range)

    deserialized = sy.deserialize(serialized)

    assert isinstance(deserialized, Range)
    assert deserialized.id == syft_range.id
    for deserialized_el, original_el in zip(deserialized, syft_range):
        assert deserialized_el == original_el


def test_range_send(client: sy.VirtualMachineClient) -> None:
    syft_range = Range(5)
    ptr = syft_range.send(client)

    assert ptr.__class__.__name__ == "RangePointer"

    res = ptr.get()
    for res_el, original_el in zip(res, syft_range):
        assert res_el == original_el
