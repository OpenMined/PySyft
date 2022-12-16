# syft absolute
import syft as sy
from syft.lib.python.tuple import Tuple
from syft.proto.lib.python.tuple_pb2 import Tuple as Tuple_PB


def test_tuple_serde() -> None:
    syft_tuple = Tuple((1, 2))

    serialized = syft_tuple._object2proto()

    assert isinstance(serialized, Tuple_PB)

    deserialized = Tuple._proto2object(proto=serialized)

    assert isinstance(deserialized, Tuple)
    for deserialized_el, original_el in zip(deserialized, syft_tuple):
        assert deserialized_el == original_el


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
