from syft.lib.tensor.tensorbase import SyftTensor, FloatTensor, DataTensor
import torch


def test_children():
    def get_children_types(t, l=None):
        l = [] if l is None else l
        if hasattr(t, "child"): return l + [type(t)] + get_children_types(t.child, l)
        else: return l + [type(t)]

    t = SyftTensor.FloatTensor([1,2,3])
    assert get_children_types(t) == [SyftTensor, FloatTensor, DataTensor, torch.Tensor]

def test_addition_floattensor():
    t1 = SyftTensor.FloatTensor([1,2,3])
    t2 = SyftTensor.FloatTensor([4,5,6])
    t3 = t1 + t2
    assert all(t3.child.child.child.numpy() == [5.0, 7.0, 9.0])