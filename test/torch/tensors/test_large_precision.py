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


def test_split_restore():
    bits = 32

    result_128 = LargePrecisionTensor._split_number(87721325272084551684339671875103718004, bits)
    result_64 = LargePrecisionTensor._split_number(4755382571665082714, bits)
    result_32 = LargePrecisionTensor._split_number(1107198784, bits)

    assert len(result_128) == 4
    assert len(result_64) == 2
    assert len(result_32) == 1

    assert (
        LargePrecisionTensor._restore_number(result_128, bits)
        == 87721325272084551684339671875103718004
    )
    assert LargePrecisionTensor._restore_number(result_64, bits) == 4755382571665082714
    assert LargePrecisionTensor._restore_number(result_32, bits) == 1107198784


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
