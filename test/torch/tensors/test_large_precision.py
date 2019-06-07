import torch
from syft.frameworks.torch.tensors.interpreters import LargePrecisionTensor


def test_wrap(workers):
    """
    Test the .on() wrap functionality for LargePrecisionTensor
    """
    x_tensor = torch.Tensor([1, 2, 3])
    x = LargePrecisionTensor().on(x_tensor)
    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, LargePrecisionTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test_fix_prec(workers):
    x = torch.tensor([1.5, 2.0, 3.0])
    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=256)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_2d_tensors(workers):
    x = torch.tensor([[1.5, 2.0, 3.0], [4.5, 5.0, 6.0]])
    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=256)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_3d_tensors(workers):
    x = torch.tensor([[[1.5, 2.0, 3.0]], [[4.5, 5.0, 6.0]], [[7.0, 8.0, 9.0]]])
    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=256)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_negative_numbers(workers):
    x = torch.tensor([[[-1.5, 2.0, 3.0]], [[4.5, 5.0, 6.0]], [[7.0, 8.0, 9.0]]])
    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=256)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_add_multiple_dimensions(workers):
    x = torch.tensor([[[-1.5, -2.0, -3.0]], [[4.5, 5.0, -3.0]]])
    y = torch.tensor([[[-1.5, -2.0, -3.0]], [[4.5, 5.0, 6.0]]])
    lpt1 = x.fix_prec(internal_type=torch.int16, precision_fractional=256)
    lpt2 = y.fix_prec(internal_type=torch.int16, precision_fractional=256)
    expected = torch.tensor([[[-3.0, -4.0, -6.0]], [[9.0, 10.0, 3.0]]])
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_subtract():
    internal_type = torch.int16
    precision_fractional = 256
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([-90000000000000010.0])
    expected = torch.tensor([-90000000000000000.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_add():
    internal_type = torch.int16
    precision_fractional = 256
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([30.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_add_different_dims():
    internal_type = torch.int16
    precision_fractional = 256
    x1 = torch.tensor([100000.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([100020.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))
