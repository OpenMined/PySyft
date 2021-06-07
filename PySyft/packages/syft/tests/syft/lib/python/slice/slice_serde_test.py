# third party
import torch as th

# syft absolute
import syft as sy
from syft.lib.python.slice import Slice
from syft.proto.lib.python.slice_pb2 import Slice as Slice_PB


def test_slice_serde() -> None:
    syft_slice = Slice(1, 3, -1)
    serialized = syft_slice._object2proto()

    assert isinstance(serialized, Slice_PB)

    deserialized = Slice._proto2object(proto=serialized)

    assert isinstance(deserialized, Slice)
    assert deserialized.id == syft_slice.id
    assert deserialized.start == syft_slice.start
    assert deserialized.stop == syft_slice.stop
    assert deserialized.step == syft_slice.step


def test_slice_send(client: sy.VirtualMachineClient) -> None:
    syft_slice = Slice(1, 3, None)
    ptr = syft_slice.send(client)

    # Check pointer type
    assert ptr.__class__.__name__ == "SlicePointer"

    # Check that we can get back the object
    res = ptr.get()
    assert res.start == syft_slice.start
    assert res.stop == syft_slice.stop
    assert res.step == syft_slice.step


def test_slice_tensor(client) -> None:
    syft_slice = Slice(0, 1)
    slice_ptr = syft_slice.send(client)

    t = th.Tensor([1, 2, 3])
    t_ptr = t.send(client)
    res_ptr = t_ptr[slice_ptr]

    # Check that we can get back the object
    res = res_ptr.get()
    assert res == t[0:1]

    res_ptr2 = t_ptr[0:1]
    res2 = res_ptr2.get()

    assert res == res2

    last_ptr = t_ptr[-1]
    last = last_ptr.item().get()
    assert last == 3
