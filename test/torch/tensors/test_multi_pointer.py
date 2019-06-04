import pytest
import torch as th
import syft as sy

from syft.frameworks.torch.tensors.decorators import LoggingTensor


def test_multi_pointers(workers):
    """
    Ensure that the sy.combine_pointers works as expected
    """

    bob = workers["bob"]
    alice = workers["alice"]

    a = th.tensor([1, 2, 3, 4, 5]).send(bob, alice)

    b = a + a

    c = b.get(sum_results=True)
    assert (c == th.tensor([4, 8, 12, 16, 20])).all()

    b = a + a
    c = b.get(sum_results=False)
    assert len(c) == 2
    assert (c[0] == th.tensor([2, 4, 6, 8, 10])).all

    # test default sum pointer state
    b = a + a
    c = b.get()
    assert len(c) == 2
    assert (c[0] == th.tensor([2, 4, 6, 8, 10])).all
