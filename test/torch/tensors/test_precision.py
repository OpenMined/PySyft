import torch

from syft.frameworks.torch.tensors.interpreters import FixedPrecisionTensor


def test_wrap(workers):
    """
    Test the .on() wrap functionality for LoggingTensor
    """

    x_tensor = torch.Tensor([1, 2, 3])
    x = FixedPrecisionTensor().on(x_tensor)
    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, FixedPrecisionTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test_encode_decode(workers):

    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()
    assert x.child.child[0] == 100
    x = x.float_prec()

    assert x[0] == 0.1


def test_add_method():

    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    y = x + x

    assert y.child.child[0] == 200
    y = y.float_prec()

    assert y[0] == 0.2


def test_add_func():

    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    y = torch.add(x, x)

    assert y.child.child[0] == 200
    y = y.float_prec()

    assert y[0] == 0.2


def test_fixed_precision_and_sharing(workers):

    bob, alice = (workers["bob"], workers["alice"])

    x = torch.tensor([1, 2, 3, 4.0]).fix_prec().share(bob, alice)
    out = x.get().float_prec()

    assert out[0] == 1

    x = torch.tensor([1, 2, 3, 4.0]).fix_prec().share(bob, alice)

    y = x + x

    y = y.get().float_prec()
    assert y[0] == 2
