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


def test_large_prec(workers):
    x = torch.tensor([1.5, 2.0, 3.0])
    enlarged = x.large_prec(precision=16, virtual_prec=256)
    restored = enlarged.restore_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_2d_tensors(workers):
    x = torch.tensor([[1.5, 2.0, 3.0], [4.5, 5.0, 6.0]])
    enlarged = x.large_prec(precision=16, virtual_prec=256)
    restored = enlarged.restore_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_3d_tensors(workers):
    x = torch.tensor([[[1.5, 2.0, 3.0]], [[4.5, 5.0, 6.0]], [[7., 8.0, 9.0]]])
    enlarged = x.large_prec(precision=16, virtual_prec=256)
    restored = enlarged.restore_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_add():
    bits = 16
    virtual_prec = 256
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([30.0])
    lpt1 = x1.large_prec(precision=bits, virtual_prec=virtual_prec)
    lpt2 = x2.large_prec(precision=bits, virtual_prec=virtual_prec)
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.restore_precision()))


def test_add_different_dims():
    bits = 16
    virtual_prec = 256
    x1 = torch.tensor([100000.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([100020.0])
    lpt1 = x1.large_prec(precision=bits, virtual_prec=virtual_prec)
    lpt2 = x2.large_prec(precision=bits, virtual_prec=virtual_prec)
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.restore_precision()))
