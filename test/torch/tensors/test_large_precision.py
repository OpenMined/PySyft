import pytest

import torch

from syft.frameworks.torch.tensors.interpreters.large_precision import LargePrecisionTensor


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
    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_2d_tensors(workers):
    x = torch.tensor([[1.5, 2.0, 3.0], [4.5, 5.0, 6.0]])
    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_3d_tensors(workers):
    x = torch.tensor([[[1.5, 2.0, 3.0]], [[4.5, 5.0, 6.0]], [[7.0, 8.0, 9.0]]])
    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_negative_numbers(workers):
    x = torch.tensor([[[-1.5, 2.0, 3.0]], [[4.5, 5.0, 6.0]], [[7.0, 8.0, 9.0]]])
    enlarged = x.fix_prec(
        base=10, internal_type=torch.int16, precision_fractional=128, verbose=True
    )
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_add_multiple_dimensions(workers):
    x = torch.tensor([[[-1.5, -2.0, -3.0]], [[4.5, 5.0, -3.0]]])
    y = torch.tensor([[[-1.5, -2.0, -3.0]], [[4.5, 5.0, 6.0]]])
    lpt1 = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
    lpt2 = y.fix_prec(internal_type=torch.int16, precision_fractional=128)
    expected = torch.tensor([[[-3.0, -4.0, -6.0]], [[9.0, 10.0, 3.0]]])
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_add_negative_values():
    internal_type = torch.int16
    precision_fractional = 128
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([-90000000000000010.0])
    expected = torch.tensor([-90000000000000000.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_add():
    internal_type = torch.int16
    precision_fractional = 128
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([30.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_iadd():
    internal_type = torch.int16
    precision_fractional = 128
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([30.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt1.add_(lpt2)
    assert torch.all(torch.eq(expected, lpt1.float_precision()))


def test_add_different_dims():
    internal_type = torch.int16
    precision_fractional = 128
    x1 = torch.tensor([100000.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([100020.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 + lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_mul():
    internal_type = torch.int16
    precision_fractional = 32
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([200.0])
    expected.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 * lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_imul():
    internal_type = torch.int16
    precision_fractional = 32
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([20.0])
    expected = torch.tensor([200.0])
    expected.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt1.mul_(lpt2)
    assert torch.all(torch.eq(expected, lpt1.float_precision()))


def test_mul_multiple_dims():
    internal_type = torch.int16
    precision_fractional = 32
    x = torch.tensor([[[-1.5, -2.0, -3.0]], [[4.0, 5.0, -3.0]]])
    y = torch.tensor([[[-1.5, -2.0, -3.0]], [[4.0, 5.0, 6.0]]])
    expected = torch.tensor([[[2.25, 4.0, 9.0]], [[16.0, 25.0, -18.0]]])
    expected.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt1 = x.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = y.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 * lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_concat_ops():
    internal_type = torch.int16
    precision_fractional = 32
    x = torch.tensor([[[-1.5, -2.0, -3.0]], [[4.0, 5.0, -3.0]]])
    y = torch.tensor([[[-1.5, -2.0, -3.0]], [[4.0, 5.0, 6.0]]])
    z = torch.tensor([[[-1.0, -2.0, 2.5]], [[4.0, -5.0, 7.5]]])
    expected = torch.tensor([[[1.25, 2.0, 11.5]], [[20.0, 20.0, -10.5]]])
    expected.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt_x = x.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt_y = y.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt_z = z.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = (lpt_x * lpt_y) + lpt_z

    assert torch.all(torch.eq(expected, result.float_precision()))


def test_uint8_representation(workers):
    x = torch.tensor([[1.5, 2.0, 3.0], [4.5, 5.0, 6.0]])
    enlarged = x.fix_prec(internal_type=torch.uint8, precision_fractional=128)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


def test_sub():
    internal_type = torch.int16
    precision_fractional = 128
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([90000000000000010.0])
    expected = torch.tensor([-90000000000000000.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 - lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


def test_isub():
    internal_type = torch.int16
    precision_fractional = 128
    x1 = torch.tensor([10.0])
    x2 = torch.tensor([90000000000000010.0])
    expected = torch.tensor([-90000000000000000.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt1.sub_(lpt2)
    assert torch.all(torch.eq(expected, lpt1.float_precision()))


def test_diff_dims_in_same_tensor():
    internal_type = torch.int16
    precision_fractional = 128
    x = torch.tensor([2000.0, 1.0])
    lpt_x = x.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    restored = lpt_x.float_precision()
    assert torch.all(torch.eq(x, restored))


def test_mod():
    internal_type = torch.int16
    precision_fractional = 128
    expected = torch.tensor([6.0, 3.0])
    x1 = torch.tensor([6.0, 12.0])
    x2 = torch.tensor([9.0])
    lpt1 = x1.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    lpt2 = x2.fix_prec(internal_type=internal_type, precision_fractional=precision_fractional)
    result = lpt1 % lpt2
    assert torch.all(torch.eq(expected, result.float_precision()))


test_data = [
    (torch.tensor([1]), torch.tensor([1.0])),
    (torch.tensor([1.0]), torch.tensor([1.0])),
    (torch.tensor([2000.0, 1.0]), torch.tensor([2000.0, 1.0])),
    (torch.tensor([2000.0, 1]), torch.tensor([2000.0, 1.0])),
    (torch.tensor([-2000.0]), torch.tensor([-2000.0])),
    (torch.tensor([-2000.0, -50]), torch.tensor([-2000.0, -50.0])),
    (torch.tensor([-2000.0, 50]), torch.tensor([-2000.0, 50.0])),
    (
        torch.tensor([[-2000.0, 50], [1000.5, -25]]),
        torch.tensor([[-2000.0, 50.0], [1000.5, -25.0]]),
    ),
    (torch.tensor([-2000.0123458910]), torch.tensor([-2000.0123458910])),
    (torch.tensor([2000.0123458910]), torch.tensor([2000.0123458910])),
]


@pytest.mark.parametrize("x, expected", test_data)
def test_types(x, expected):
    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(expected, restored))


@pytest.mark.parametrize("x, expected", test_data)
def test_share_and_get(workers, x, expected):
    alice, bob, james = (workers["alice"], workers["bob"], workers["james"])

    enlarged = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
    expected = expected.fix_prec(internal_type=torch.int16, precision_fractional=128)

    shared = enlarged.share(alice, bob, crypto_provider=james)

    result = shared.get().float_precision()

    assert (expected.float_precision() == result).all()


def test_share_add(workers):
    alice, bob, james = (workers["alice"], workers["bob"], workers["james"])

    x = torch.tensor([5.0])
    expected = x + x

    x = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
    x_shared = x.share(alice, bob, crypto_provider=james)

    y = (x_shared + x_shared).get().float_precision()

    assert torch.all(torch.eq(expected, y))


def test_share_sub(workers):
    alice, bob, james = (workers["alice"], workers["bob"], workers["james"])

    x = torch.tensor([5.0])
    expected = x - x

    x = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
    x_shared = x.share(alice, bob, crypto_provider=james)

    y = (x_shared - x_shared).get().float_precision()

    assert torch.all(torch.eq(expected, y))


def test_storage():
    x = torch.tensor([1.0, 2.0, 3.0])
    enlarged = x.fix_prec(storage="large")
    restored = enlarged.float_precision()
    # And now x and restored must be the same
    assert torch.all(torch.eq(x, restored))


# def test_share_mul(workers):
#     alice, bob, james = (workers["alice"], workers["bob"], workers["james"])
#
#     x = torch.tensor([5.0])
#     expected = x * x
#
#     x = x.fix_prec(internal_type=torch.int16, precision_fractional=128)
#     x_shared = x.share(alice, bob, crypto_provider=james)
#
#     # TODO
#     #  This is tricky. The multiplication of the shares in the workers will not yield the expected results
#     #  because they have the original number decomposed into parts.
#     y = (x_shared * x_shared).get().float_precision()
#
#     assert torch.all(torch.eq(expected, y))
