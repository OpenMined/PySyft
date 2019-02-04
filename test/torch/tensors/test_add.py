import random
import torch
import syft

from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor


def test_wrap(workers):
    """
    Test the .on() wrap functionality for LoggingTensor
    """

    x_tensor = torch.Tensor([1, 2, 3])
    x = AdditiveSharingTensor().on(x_tensor)
    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, AdditiveSharingTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test_encode_decode(workers):

    x = torch.tensor([1, 2, 3]).share(workers["bob"], workers["alice"], workers["james"])

    x = x.get()

    assert x[0] == 1


def test_add(workers):

    x = torch.tensor([1, 2, 3]).share(workers["bob"], workers["alice"], workers["james"])

    y = (x + x).get()

    assert y[0] == 2


def test_sub(workers):

    x = torch.tensor([1, 2, 3]).share(workers["bob"], workers["alice"], workers["james"])

    y = (x - x).get()

    assert y[0] == 0


def test_fixed_precision_and_sharing(workers):

    bob, alice = (workers["bob"], workers["alice"])

    x = torch.tensor([1, 2, 3, 4.0]).fix_prec().share(bob, alice)
    out = x.get().float_prec()

    assert out[0] == 1

    x = torch.tensor([1, 2, 3, 4.0]).fix_prec().share(bob, alice)

    y = x + x

    y = y.get().float_prec()
    assert y[0] == 2
