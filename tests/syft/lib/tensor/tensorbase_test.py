from syft.lib.tensor.tensorbase import SyftTensor, FloatTensor, DataTensor
from typing import List, Any
import torch


def test_children() -> None:
    def get_children_types(t: Any, types: Any = None) -> List[Any]:
        types = [] if types is None else types
        if hasattr(t, "child"):
            return types + [type(t)] + get_children_types(t.child, types)
        else:
            return types + [type(t)]

    t = SyftTensor.FloatTensor([1, 2, 3])
    assert get_children_types(t) == [SyftTensor, FloatTensor, DataTensor, torch.Tensor]


def test_addition_floattensor() -> None:
    t1 = SyftTensor.FloatTensor([1, 2, 3])
    t2 = SyftTensor.FloatTensor([4, 5, 6])
    t3 = t1 + t2
    assert all(t3.child.child.child.numpy() == [5.0, 7.0, 9.0])
