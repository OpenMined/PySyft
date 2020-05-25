import torch as th
import syft as sy

from syft.generic.pointers.multi_pointer import MultiPointerTensor


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


def test_dim(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    a = th.tensor([1, 2, 3, 4, 5]).send(bob, alice)

    assert a.dim() == 1


def test_simplify(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    a = th.tensor([1, 2, 3, 4, 5]).send(bob, alice)
    ser = sy.serde.serialize(a)
    detail = sy.serde.deserialize(ser).child
    assert isinstance(detail, MultiPointerTensor)
    for key in a.child.child:
        assert key in detail.child
