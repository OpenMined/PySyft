# third party
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.lib.python.dict import Dict
from syft.proto.lib.python.dict_pb2 import Dict as Dict_PB


def test_dict_serde() -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = Dict({"t1": t1, "t2": t2})
    assert type(getattr(syft_list, "id", None)) is UID

    serialized = syft_list._object2proto()

    assert isinstance(serialized, Dict_PB)

    deserialized = Dict._proto2object(proto=serialized)

    assert isinstance(deserialized, Dict)
    assert deserialized.id == syft_list.id
    for deserialized_el, original_el in zip(deserialized, syft_list):
        assert deserialized_el == original_el


def test_list_send() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = Dict({"t1": t1, "t2": t2})
    ptr = syft_list.send(alice_client)
    # Check pointer type
    assert ptr.__class__.__name__ == "DictPointer"

    # Check that we can get back the object
    res = ptr.get()
    for res_el, original_el in zip(res, syft_list):
        assert res_el == original_el
