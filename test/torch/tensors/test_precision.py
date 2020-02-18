import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor


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


def test_fix_prec_registration(hook):
    with hook.local_worker.registration_enabled():
        x = torch.tensor([1.0])
        x_fpt = x.fix_precision()

        assert hook.local_worker.get_obj(x.id) == x


def test_inplace_encode_decode(workers):

    x = torch.tensor([0.1, 0.2, 0.3])
    x.fix_prec_()
    assert (x.child.child == torch.LongTensor([100, 200, 300])).all()
    x.float_prec_()

    assert (x == torch.tensor([0.1, 0.2, 0.3])).all()

    x = torch.tensor([3.0]).fix_precision()
    assert x.float_prec_().is_wrapper is False


def test_fix_prec_inplace_registration(hook):

    with hook.local_worker.registration_enabled():
        x = torch.tensor([1.0])
        x.fix_precision_()
        assert hook.local_worker.get_obj(x.id) == torch.tensor([1.0]).fix_precision()


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


def test_torch_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    # Method syntax
    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    y = x + x

    assert (y.child.child == torch.LongTensor([200, 400, 600])).all()
    y = y.float_prec()

    assert (y == torch.tensor([0.2, 0.4, 0.6])).all()

    # Function syntax
    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    y = torch.add(x, x)

    assert (y.child.child == torch.LongTensor([200, 400, 600])).all()
    y = y.float_prec()

    assert (y == torch.tensor([0.2, 0.4, 0.6])).all()

    # With negative numbers
    x = torch.tensor([-0.1, -0.2, 0.3]).fix_prec()
    y = torch.tensor([0.4, -0.5, -0.6]).fix_prec()

    z = torch.add(x, y).float_prec()

    assert (z == torch.tensor([0.3, -0.7, -0.3])).all()

    # When overflow occurs
    x = torch.tensor([10.0, 20.0, 30.0]).fix_prec(field=1e4, precision_fractional=2)
    y = torch.add(x, x)
    y = torch.add(y, y).float_prec()

    assert (y == torch.tensor([40.0, -20.0, 20.0])).all()

    # with AdditiveSharingTensor
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec()
    y = t.fix_prec().share(bob, alice, crypto_provider=james)

    z = torch.add(x, y).get().float_prec()
    assert (z == torch.add(t, t)).all()

    z = torch.add(y, x).get().float_prec()
    assert (z == torch.add(t, t)).all()

    # with constant integer
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec()
    c = 4

    z = (x + c).float_prec()
    assert (z == (t + c)).all()

    z = (c + x).float_prec()
    assert (z == (c + t)).all()

    # with constant float
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec()
    c = 4.2

    z = (x + c).float_prec()
    assert ((z - (t + c)) < 10e-3).all()

    z = (c + x).float_prec()
    assert ((z - (c + t)) < 10e-3).all()


def test_torch_add_():
    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    y = x.add_(x)

    assert (y.child.child == torch.LongTensor([200, 400, 600])).all()
    y = y.float_prec()

    assert (y == torch.tensor([0.2, 0.4, 0.6])).all()

    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()
    lr = torch.tensor(0.5).fix_prec()

    y = x.add_(lr, x)

    assert (y.child.child == torch.LongTensor([150, 300, 450])).all()
    y = y.float_prec()

    assert (y == torch.tensor([0.15, 0.3, 0.45])).all()


def test_torch_sub(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    x = torch.tensor([0.5, 0.8, 1.3]).fix_prec()
    y = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    z = torch.sub(x, y)

    assert (z.child.child == torch.LongTensor([400, 600, 1000])).all()
    z = z.float_prec()

    assert (z == torch.tensor([0.4, 0.6, 1.0])).all()

    # with AdditiveSharingTensor
    tx = torch.tensor([1.0, -2.0, 3.0])
    ty = torch.tensor([0.1, 0.2, 0.3])
    x = tx.fix_prec()
    y = ty.fix_prec().share(bob, alice, crypto_provider=james)

    z1 = torch.sub(y, x).get().float_prec()
    z2 = torch.sub(x, y).get().float_prec()

    assert (z1 == torch.sub(ty, tx)).all()
    assert (z2 == torch.sub(tx, ty)).all()

    # with constant integer
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec()
    c = 4

    z = (x - c).float_prec()
    assert (z == (t - c)).all()

    z = (c - x).float_prec()
    assert (z == (c - t)).all()

    # with constant float
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec()
    c = 4.2

    z = (x - c).float_prec()
    assert ((z - (t - c)) < 10e-3).all()

    z = (c - x).float_prec()
    assert ((z - (c - t)) < 10e-3).all()


def test_torch_sub_():
    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    y = x.sub_(x)

    assert (y.child.child == torch.LongTensor([0, 0, 0])).all()
    y = y.float_prec()

    assert (y == torch.tensor([0, 0, 0.0])).all()

    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()
    lr = torch.tensor(0.5).fix_prec()

    y = x.sub_(lr, x)

    assert (y.child.child == torch.LongTensor([50, 100, 150])).all()
    y = y.float_prec()

    assert (y == torch.tensor([0.05, 0.1, 0.15])).all()


def test_torch_mul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    # mul with non standard fix precision
    x = torch.tensor([2.113]).fix_prec(precision_fractional=2)

    y = torch.mul(x, x)

    assert y.child.child == torch.LongTensor([445])
    assert y.child.precision_fractional == 2

    y = y.float_prec()

    assert y == torch.tensor([4.45])

    # Mul with negative numbers
    x = torch.tensor([2.113]).fix_prec()
    y = torch.tensor([-0.113]).fix_prec()

    z = torch.mul(x, y)

    assert z.child.precision_fractional == 3

    z = z.float_prec()
    assert z == torch.tensor([-0.2380])

    x = torch.tensor([11.0]).fix_prec(field=2 ** 16, precision_fractional=2)
    y = torch.mul(x, x).float_prec()

    assert y == torch.tensor([121.0])

    # mixing + and *
    x = torch.tensor([2.113]).fix_prec()
    y = torch.tensor([-0.113]).fix_prec()
    z = torch.mul(x, y + y)

    assert z.child.precision_fractional == 3

    z = z.float_prec()

    assert z == torch.tensor([-0.4770])

    # with AST
    t = torch.tensor([1.0, -2.0, 3.0])
    u = torch.tensor([1.0, -2.0, -3.0])
    x = t.fix_prec()
    y = u.fix_prec().share(bob, alice, crypto_provider=james)

    z = torch.mul(x, y).get().float_prec()

    assert (z == torch.mul(t, u)).all()


def test_torch_div(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    # With scalar
    x = torch.tensor([[9.0, 25.42], [3.3, 0.0]]).fix_prec()
    y = torch.tensor([[3.0, 6.2], [3.3, 4.7]]).fix_prec()

    z = torch.div(x, y).float_prec()

    assert (z == torch.tensor([[3.0, 4.1], [1.0, 0.0]])).all()

    # With negative numbers
    x = torch.tensor([[-9.0, 25.42], [-3.3, 0.0]]).fix_prec()
    y = torch.tensor([[3.0, -6.2], [-3.3, 4.7]]).fix_prec()

    z = torch.div(x, y).float_prec()

    assert (z == torch.tensor([[-3.0, -4.1], [1.0, 0.0]])).all()

    # AST divided by FPT
    x = torch.tensor([[9.0, 25.42], [3.3, 0.0]]).fix_prec().share(bob, alice, crypto_provider=james)
    y = torch.tensor([[3.0, 6.2], [3.3, 4.7]]).fix_prec()

    z = torch.div(x, y).get().float_prec()

    assert (z == torch.tensor([[3.0, 4.1], [1.0, 0.0]])).all()


def test_inplace_operations():
    a = torch.tensor([5.0, 6.0]).fix_prec()
    b = torch.tensor([2.0]).fix_prec()

    a /= b
    assert (a.float_prec() == torch.tensor([2.5, 3.0])).all()

    a *= b
    assert (a.float_prec() == torch.tensor([5.0, 6.0])).all()

    a += b
    assert (a.float_prec() == torch.tensor([7.0, 8.0])).all()

    a -= b
    assert (a.float_prec() == torch.tensor([5.0, 6.0])).all()


def test_torch_pow():

    m = torch.tensor([[1, 2], [3, 4.0]])
    x = m.fix_prec()
    y = (x ** 3).float_prec()

    assert (y == (m ** 3)).all()


def test_torch_matmul(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    m = torch.tensor([[1, 2], [3, 4.0]])
    x = m.fix_prec()
    y = torch.matmul(x, x).float_prec()

    assert (y == torch.matmul(m, m)).all()

    # with AST
    m = torch.tensor([[1, 2], [3, 4.0]])
    x = m.fix_prec()
    y = m.fix_prec().share(bob, alice, crypto_provider=james)

    z = (x @ y).get().float_prec()

    assert (z == torch.matmul(m, m)).all()


def test_torch_addmm():
    weight = nn.Parameter(torch.tensor([[1.0, 2], [4.0, 2]])).fix_precision()
    inputs = nn.Parameter(torch.tensor([[1.0, 2]])).fix_precision()
    bias = nn.Parameter(torch.tensor([1.0, 2])).fix_precision()

    fp_result = torch.addmm(bias, inputs, weight)

    assert (fp_result.float_precision() == torch.tensor([[10.0, 8.0]])).all()


def test_torch_dot(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).fix_prec()
    y = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0]).fix_prec()

    assert torch.dot(x, y).float_prec() == 45


def test_torch_inverse_approx(workers):
    """
    Test the approximate inverse with different tolerance depending on
    the precision_fractional considered
    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    fix_prec_tolerance = {3: 1 / 100, 4: 1 / 100, 5: 1 / 100}

    for prec_frac, tolerance in fix_prec_tolerance.items():
        for t in [
            torch.tensor([[0.4, -0.1], [-0.4, 2.0]]),
            torch.tensor([[1, -0.6], [0.4, 4.0]]),
            torch.tensor([[1, 0.2], [0.4, 4.0]]),
        ]:
            t_sh = t.fix_precision(precision_fractional=prec_frac).share(
                alice, bob, crypto_provider=james
            )
            r_sh = t_sh.inverse()
            r = r_sh.get().float_prec()
            t = t.inverse()
            diff = (r - t).abs().max()
            norm = (r + t).abs().max() / 2

            assert (diff / (tolerance * norm)) < 1


def test_torch_exp_approx(workers):
    """
    Test the approximate exponential with different tolerance depending on
    the precision_fractional considered
    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    fix_prec_tolerance = {3: 20 / 100, 4: 5 / 100, 5: 5 / 100}

    for prec_frac, tolerance in fix_prec_tolerance.items():
        cumsum = torch.zeros(5)
        for i in range(10):
            t = torch.tensor([0.0, 1, 2, 3, 4])
            t_sh = t.fix_precision(precision_fractional=prec_frac).share(
                alice, bob, crypto_provider=james
            )
            r_sh = t_sh.exp()
            r = r_sh.get().float_prec()
            t = t.exp()
            diff = (r - t).abs()
            norm = (r + t) / 2
            cumsum += diff / (tolerance * norm)

        cumsum /= 10
        assert (cumsum < 1).all()


def test_torch_sigmoid_approx(workers):
    """
    Test the approximate sigmoid with different tolerance depending on
    the precision_fractional considered
    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    fix_prec_tolerance_by_method = {
        "exp": {3: 6 / 100, 4: 1 / 100, 5: 1 / 100},
        "maclaurin": {3: 7 / 100, 4: 15 / 100, 5: 15 / 100},
    }

    for method, fix_prec_tolerance in fix_prec_tolerance_by_method.items():
        for prec_frac, tolerance in fix_prec_tolerance.items():
            t = torch.tensor(range(-10, 10)) * 0.5
            t_sh = t.fix_precision(precision_fractional=prec_frac).share(
                alice, bob, crypto_provider=james
            )
            r_sh = t_sh.sigmoid(method=method)
            r = r_sh.get().float_prec()
            t = t.sigmoid()
            diff = (r - t).abs().max()
            norm = (r + t).abs().max() / 2

            assert (diff / (tolerance * norm)) < 1


def test_torch_log_approx(workers):
    """
    Test the approximate logarithm with different tolerance depending on
    the precision_fractional considered
    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    fix_prec_tolerance = {3: 100 / 100, 4: 3 / 100, 5: 2 / 100}

    for prec_frac, tolerance in fix_prec_tolerance.items():
        cumsum = torch.zeros(9)
        for i in range(10):
            t = torch.tensor([0.1, 0.5, 2, 5, 10, 20, 50, 100, 250])
            t_sh = t.fix_precision(precision_fractional=prec_frac).share(
                alice, bob, crypto_provider=james
            )
            r_sh = t_sh.log()
            r = r_sh.get().float_prec()
            t = t.log()
            diff = (r - t).abs()
            norm = (r + t) / 2
            cumsum += diff / (tolerance * norm)

        cumsum /= 10
        assert (cumsum.abs() < 1).all()


def test_torch_conv2d(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    im = torch.Tensor(
        [
            [
                [[0.5, 1.0, 2.0], [3.5, 4.0, 5.0], [6.0, 7.5, 8.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.5, 15.0], [16.0, 17.5, 18.0]],
            ]
        ]
    )
    w = torch.Tensor(
        [
            [[[0.0, 3.0], [1.5, 1.0]], [[2.0, 2.0], [2.5, 2.0]]],
            [[[-0.5, -1.0], [-2.0, -1.5]], [[0.0, 0.0], [0.0, 0.5]]],
        ]
    )
    bias = torch.Tensor([-1.3, 15.0])

    im_fp = im.fix_precision()
    w_fp = w.fix_precision()
    bias_fp = bias.fix_precision()

    res0 = torch.conv2d(im_fp, w_fp, bias=bias_fp, stride=1).float_precision()
    res1 = torch.conv2d(
        im_fp, w_fp[:, 0:1].contiguous(), bias=bias_fp, stride=2, padding=3, dilation=2, groups=2
    ).float_precision()

    expected0 = torch.conv2d(im, w, bias=bias, stride=1)
    expected1 = torch.conv2d(
        im, w[:, 0:1].contiguous(), bias=bias, stride=2, padding=3, dilation=2, groups=2
    )

    assert (res0 == expected0).all()
    assert (res1 == expected1).all()


def test_torch_nn_functional_linear():
    tensor = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec()
    weight = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec()

    result = F.linear(tensor, weight).float_prec()

    expected = torch.tensor([[5.0, 11.0], [11.0, 25.0]])

    assert (result == expected).all()

    tensor = nn.Parameter(torch.tensor([[1.0, -2], [3, 4]])).fix_prec()
    weight = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec()

    result = F.linear(tensor, weight).float_prec()

    expected = torch.tensor([[-3.0, -5], [11.0, 25.0]])

    assert (result == expected).all()

    tensor = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec(precision_fractional=2)
    weight = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]])).fix_prec(precision_fractional=2)

    result = F.linear(tensor, weight).float_prec()

    expected = torch.tensor([[5.0, 11.0], [11.0, 25.0]])

    assert (result == expected).all()


def test_operate_with_integer_constants():
    x = torch.tensor([1.0])
    x_fp = x.fix_precision()

    r_fp = x_fp + 10
    r = r_fp.float_precision()
    assert r == x + 10

    r_fp = x_fp - 7
    r = r_fp.float_precision()
    assert r == x - 7

    r_fp = x_fp * 2
    assert r_fp.float_precision() == x * 2

    r_fp = x_fp / 5
    assert r_fp.float_precision() == x / 5


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


def test_comp():
    x = torch.tensor([3.1]).fix_prec()
    y = torch.tensor([3.1]).fix_prec()

    assert (x >= y).float_prec()
    assert (x <= y).float_prec()
    assert not (x > y).float_prec()
    assert not (x < y).float_prec()

    x = torch.tensor([3.1]).fix_prec()
    y = torch.tensor([2.1]).fix_prec()

    assert (x >= y).float_prec()
    assert not (x <= y).float_prec()
    assert (x > y).float_prec()
    assert not (x < y).float_prec()

    x = torch.tensor([2.1]).fix_prec()
    y = torch.tensor([3.1]).fix_prec()

    assert not (x >= y).float_prec()
    assert (x <= y).float_prec()
    assert not (x > y).float_prec()
    assert (x < y).float_prec()
