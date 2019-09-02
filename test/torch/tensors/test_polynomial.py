import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from syft.frameworks.torch.tensors.interpreters import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters import PolynomialTensor
import syft as sy


def test_wrap():

    x_tensor = torch.Tensor([1, 2, 3])
    x = PolynomialTensor().on(x_tensor)
    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, PolynomialTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test_sigmoid_torch():

    # Test for interpolation
    x = torch.tensor([-10.0, 4.0, 8.0, -8.0]).poly()
    result = x.sigmoid()
    expected = torch.tensor([0.0505, 0.9957, 1.0023, -0.0023])
    assert torch.allclose(result.child.child, expected, atol=1e-01)

    # Test for Taylor series
    x = torch.tensor([-2.0, -1.5, -1, 0, 0.5, 1.0]).poly(method="taylor")
    result = x.sigmoid()
    expected = torch.tensor(
        [-1.9330e02, -1.3042e01, -2.6894e-01, 2.1357e-05, 1.7265e-03, 7.3106e-01]
    )
    assert torch.allclose(result.child.child, expected, atol=1e-01)

    # Test if the tensor is hooked correctly
    assert isinstance(result.child, PolynomialTensor)
    assert isinstance(result.child.child, torch.Tensor)


def test_tanh():

    # Test for interpolation
    x = torch.tensor([-10.0, 0.5, 2.0, -8.0]).poly()
    result = x.tanh()
    expected = torch.tensor([-1.0558, 0.2851, 0.9433, -1.0585])
    assert torch.allclose(result.child.child, expected, atol=1e-00)

    # Test if the tensor is hooked correctly
    assert isinstance(result.child, PolynomialTensor)
    assert isinstance(result.child.child, torch.Tensor)


def test_exp_torch():

    x = torch.tensor([1, 4.0, 8.0, 10.0]).poly()
    result = x.exp()
    expected = torch.tensor([1.1302e01, 4.1313e01, 2.9646e03, 2.1985e04])
    assert torch.allclose(result.child.child, expected, atol=1e-01)

    # Test for Taylor Series
    x = torch.tensor([1, 4.0, 8.0, 10.0]).poly(method="taylor")
    result = x.exp()
    expected = torch.tensor([1.7167e00, 2.9084e02, 4.3630e03, 1.0517e04])
    assert torch.allclose(result.child.child, expected, atol=1e-01)

    # Test if the tensor is hooked correctly
    assert isinstance(result.child, PolynomialTensor)
    assert isinstance(result.child.child, torch.Tensor)


def test_exp_fixprecision():

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).fix_precision().poly()
    result = x.exp()
    expected = torch.tensor([11302, 19502, 22736, 57406, 207530, 598502, 1362052, 2562326])
    assert torch.allclose(result.child.child.child, expected, atol=1e-01)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).fix_precision().poly()
    result = x.exp()
    expected = torch.tensor([11302, 19502, 22736, 57406, 207530, 598502, 1362052, 2562326])
    assert torch.allclose(result.child.child.child, expected, atol=1e-01)

    # Test if the tensor is hooked correctly
    assert isinstance(result.child, PolynomialTensor)
    assert isinstance(result.child.child, FixedPrecisionTensor)
    assert isinstance(result.child.child.child, torch.Tensor)


def test_sigmoid_fixprecision():

    x = (
        torch.tensor([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        .fix_precision()
        .poly(method="interpolation")
    )

    result = x.sigmoid()
    expected = torch.tensor([134, 205, 293, 394, 500, 606, 707, 795, 866, 929, 848])
    assert torch.allclose(result.child.child.child, expected, atol=1e01)

    # Test if the tensor is hooked correctly
    assert isinstance(result.child, PolynomialTensor)
    assert isinstance(result.child.child, FixedPrecisionTensor)
    assert isinstance(result.child.child.child, torch.Tensor)


def test_tanh_fixprecision():

    y = (
        torch.tensor([-2.4, -2.3, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.6])
        .fix_precision()
        .poly(method="interpolation")
    )
    result = y.tanh()
    expected = torch.tensor(
        [
            4611686018427386920,
            4611686018427386929,
            4611686018427386982,
            4611686018427387136,
            4611686018427387356,
            4611686018427387619,
            0,
            285,
            548,
            768,
            922,
            989,
            991,
        ]
    )

    assert torch.allclose(result.child.child.child, expected, atol=1e01)

    # Test if the tensor is hooked correctly
    assert isinstance(result.child, PolynomialTensor)
    assert isinstance(result.child.child, FixedPrecisionTensor)
    assert isinstance(result.child.child.child, torch.Tensor)


def test_exp_additiveshared():

    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    james = sy.VirtualWorker(hook, id="james")

    result = (
        torch.tensor([3, 5, 4, 1])
        .fix_precision()
        .share(alice, bob, crypto_provider=james)
        .poly()
        .exp()
    )
    expected = torch.tensor([22736, 207530, 57406, 11302])

    assert torch.allclose(result.child.child.child, expected, atol=1e01)


def test_tanh_additiveshared():

    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    james = sy.VirtualWorker(hook, id="james")

    result = (
        torch.tensor([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0])
        .fix_precision()
        .share(alice, bob, crypto_provider=james)
        .poly()
        .tanh()
        .get()
    )
    expected = torch.tensor([-922, -767, -548, -284, 0, 286, 548, 768, 922, 948])

    assert torch.allclose(result.child.child.child, expected, atol=1e01)


def test_sigmoid_additiveshared():

    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    james = sy.VirtualWorker(hook, id="james")

    result = (
        torch.tensor([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        .fix_precision()
        .share(alice, bob, crypto_provider=james)
        .poly()
        .exp()
        .get()
    )
    expected = torch.tensor(
        [-5354, -9775, -11044, -8653, -3130, 4133, 11302, 16697, 19502, 22736, 57406]
    )

    assert torch.allclose(result.child.child.child, expected, atol=1e01)
