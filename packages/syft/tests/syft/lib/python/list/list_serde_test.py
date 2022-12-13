# third party
# third party
import torch as th

# syft absolute
import syft as sy
from syft.lib.python.list import List
from syft.proto.lib.python.list_pb2 import List as List_PB


def test_list_serde() -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = List([t1, t2])

    serialized = syft_list._object2proto()

    assert isinstance(serialized, List_PB)

    deserialized = List._proto2object(proto=serialized)

    assert isinstance(deserialized, List)
    for deserialized_el, original_el in zip(deserialized, syft_list):
        assert (deserialized_el == original_el).all()


def test_list_send(client: sy.VirtualMachineClient) -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = List([t1, t2])
    ptr = syft_list.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "ListPointer"

    # Check that we can get back the object
    res = ptr.get()
    for res_el, original_el in zip(res, syft_list):
        assert (res_el == original_el).all()


def test_list_bytes() -> None:
    # Testing if multiple serialization of the similar object results in same bytes
    value_1 = List([1, 2, 3])
    value_2 = List([1, 2, 3])
    assert sy.serialize(value_1, to_bytes=True) == sy.serialize(value_2, to_bytes=True)
