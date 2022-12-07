# syft absolute
import syft as sy
from syft.lib.python.tuple import Tuple


def test_tuple_serde() -> None:
    syft_tuple = Tuple((1, 2))
    
    serialized = sy.serialize(syft_tuple, to_bytes=True)
    deserialized = sy.deserialize(serialized, from_bytes=True)

    assert isinstance(deserialized, Tuple)
    assert deserialized == syft_tuple


def test_tuple_send(client: sy.VirtualMachineClient) -> None:
    syft_tuple = Tuple((1, 2))
    ptr = syft_tuple.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "TuplePointer"

    # Check that we can get back the object
    res = ptr.get()
    for res_el, original_el in zip(res, syft_tuple):
        assert res_el == original_el


def test_tuple_bytes() -> None:
    # Testing if multiple serialization of the similar object results in same bytes
    value_1 = Tuple((1, 2, 3))
    value_2 = Tuple((1, 2, 3))
    assert sy.serialize(value_1, to_bytes=True) == sy.serialize(value_2, to_bytes=True)
