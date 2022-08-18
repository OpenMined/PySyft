# third party
import torch as th

# syft absolute
import syft as sy
from syft.lib.python.list import List


def test_list_serde() -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = List([t1, t2])

    serialized = sy.serialize(syft_list)

    deserialized = sy.deserialize(serialized)

    assert isinstance(deserialized, List)
    assert deserialized.id == syft_list.id
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
