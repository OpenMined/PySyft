import torch

from syft.frameworks.torch.tensors import FixedPrecisionTensor


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


def test_add(workers):

    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    y = x + x

    assert y.child.child[0] == 200
    y = y.float_prec()

    assert y[0] == 0.2
