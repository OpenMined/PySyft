import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


@pytest.mark.parametrize("parameter", [False, True])
def test_encode_decode(workers, parameter):
    x = torch.tensor([0.1, 0.2, 0.3])
    if parameter:
        x = nn.Parameter(x)
    x = x.fix_prec()
    assert (x.child.child == torch.LongTensor([100, 200, 300])).all()
    x = x.float_prec()

    assert (x == torch.tensor([0.1, 0.2, 0.3])).all()


def test_inplace_encode_decode(workers):

    x = torch.tensor([0.1, 0.2, 0.3])
    x.fix_prec_()
    assert (x.child.child == torch.LongTensor([100, 200, 300])).all()
    x.float_prec_()

    assert (x == torch.tensor([0.1, 0.2, 0.3])).all()


def test_add_method():

    t = torch.tensor([0.1, 0.2, 0.3])
    x = t.fix_prec()

    y = x + x

    assert (y.child.child == torch.LongTensor([200, 400, 600])).all()
    y = y.float_prec()

    assert (y == t + t).all()


@pytest.mark.parametrize("method", ["t", "matmul"])
@pytest.mark.parametrize("parameter", [False, True])
def test_methods_for_linear_module(method, parameter):
    """
    Test all the methods used in the F.linear functions
    """
    if parameter:
        tensor = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]]))
    else:
        tensor = torch.tensor([[1.0, 2], [3, 4]])
    fp_tensor = tensor.fix_precision()
    if method != "t":
        fp_result = getattr(fp_tensor, method)(fp_tensor)
        result = getattr(tensor, method)(tensor)
    else:
        fp_result = getattr(fp_tensor, method)()
        result = getattr(tensor, method)()

    assert (result == fp_result.float_precision()).all()


def test_torch_add():

    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    y = torch.add(x, x)

    assert (y.child.child == torch.LongTensor([200, 400, 600])).all()
    y = y.float_prec()

    assert (y == torch.tensor([0.2, 0.4, 0.6])).all()


def test_torch_mul():

    x = torch.tensor([2.113]).fix_prec(precision_fractional=2)

    y = torch.mul(x, x)

    assert y.child.child == torch.LongTensor([445])
    assert y.child.precision_fractional == 2

    y = y.float_prec()

    assert y == torch.tensor([4.45])


def test_torch_addmm():
    weight = nn.Parameter(torch.tensor([[1.0, 2], [4.0, 2]])).fix_precision()
    inputs = nn.Parameter(torch.tensor([[1.0, 2]])).fix_precision()
    bias = nn.Parameter(torch.tensor([1.0, 2])).fix_precision()

    fp_result = torch.addmm(bias, inputs, weight)

    assert (fp_result.float_precision() == torch.tensor([[10.0, 8.0]])).all()


def test_torch_nn_functional_linear():
    tensor = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec()
    weight = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec()

    result = F.linear(tensor, weight).float_prec()

    expected = torch.tensor([[5.0, 11.0], [11.0, 25.0]])

    assert (result == expected).all()

    tensor = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec(precision_fractional=2)
    weight = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec(precision_fractional=2)

    result = F.linear(tensor, weight).float_prec()

    expected = torch.tensor([[5.0, 11.0], [11.0, 25.0]])

    assert (result == expected).all()


def test_fixed_precision_and_sharing(workers):

    bob, alice = (workers["bob"], workers["alice"])

    x = torch.tensor([1, 2, 3, 4.0]).fix_prec().share(bob, alice)
    out = x.get().float_prec()

    assert (out == torch.tensor([1, 2, 3, 4.0])).all()

    x = torch.tensor([1, 2, 3, 4.0]).fix_prec().share(bob, alice)

    y = x + x

    y = y.get().float_prec()
    assert (y == torch.tensor([2, 4, 6, 8.0])).all()


def test_get_preserves_attributes(workers):
    bob, alice = (workers["bob"], workers["alice"])

    x = torch.tensor([1, 2, 3, 4.0]).fix_prec(precision_fractional=1).share(bob, alice)
    out = x.get().float_prec()

    assert (out == torch.tensor([1, 2, 3, 4.0])).all()
