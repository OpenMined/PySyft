# third party
import torch as th

# syft absolute
from syft.lib.python.list import List
from syft.proto.lib.python.list_pb2 import List as List_PB


def test_serde() -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = List([t1, t2])

    serialized = syft_list._object2proto()

    assert isinstance(serialized, List_PB)

    deserialized = List._proto2object(proto=serialized)

    assert isinstance(deserialized, List)
    assert deserialized.id == syft_list.id
    for deserialized_el, original_el in zip(deserialized, [t1, t2]):
        assert (deserialized_el == original_el).all()
