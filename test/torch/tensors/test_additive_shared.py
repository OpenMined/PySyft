import pytest
import random
import torch
import syft

from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor


def test_wrap(workers):
    """
    Test the .on() wrap functionality for AdditiveSharingTensor
    """

    x_tensor = torch.Tensor([1, 2, 3])
    x = AdditiveSharingTensor().on(x_tensor)
    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, AdditiveSharingTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test_encode_decode(workers):

    t = torch.tensor([1, 2, 3])
    x = t.share(workers["bob"], workers["alice"], workers["james"])

    x = x.get()

    assert (x == t).all()


def test_add(workers):

    t = torch.tensor([1, 2, 3])
    x = torch.tensor([1, 2, 3]).share(workers["bob"], workers["alice"], workers["james"])

    y = (x + x).get()

    assert (y == (t + t)).all()


def test_sub(workers):

    t = torch.tensor([1, 2, 3])
    x = torch.tensor([1, 2, 3]).share(workers["bob"], workers["alice"], workers["james"])

    y = (x - x).get()

    assert (y == (t - t)).all()


def test_mul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    t = torch.tensor([1, 2, 3, 4.0])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    y = (x * x).get().float_prec()

    assert (y == (t * t)).all()


def test_mul_with_no_crypto_provider(workers):
    bob, alice = (workers["bob"], workers["alice"])
    x = torch.tensor([1, 2, 3, 4.0]).fix_prec().share(bob, alice)
    with pytest.raises(AttributeError):
        y = (x * x).get().float_prec()


def test_matmul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    m = torch.tensor([[1, 2], [3, 4.0]])
    x = m.fix_prec().share(bob, alice, crypto_provider=james)
    y = (x @ x).get().float_prec()

    assert (y == (m @ m)).all()


def test_fixed_precision_and_sharing(workers):

    bob, alice = (workers["bob"], workers["alice"])

    t = torch.tensor([1, 2, 3, 4.0])
    x = t.fix_prec().share(bob, alice)
    out = x.get().float_prec()

    assert (out == t).all()

    x = t.fix_prec().share(bob, alice)

    y = x + x

    y = y.get().float_prec()
    assert (y == (t + t)).all()
