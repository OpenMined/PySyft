import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from syft.frameworks.torch.tensors.interpreters import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters import PolynomialTensor
import syft as sy


def test_wrap():

    hook = sy.TorchHook(torch)
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


def test_tanh():

    # Test for interpolation
    x = torch.tensor([-10.0, 0.5, 2.0, -8.0]).poly()
    result = x.tanh()
    expected = torch.tensor([-1.0558, 0.2851, 0.9433, -1.0585])
    assert torch.allclose(result.child.child, expected, atol=1e-00)

    # Test for Taylor series
    # Taylor series approximations for tanh give error. Needs correction.


def test_exp_torch():

    x = torch.tensor([1, 4.0, 8.0, 10.0]).poly()
    result = x.exp()
    expected = torch.tensor([1.1302e01, 4.1313e01, 2.9646e03, 2.1985e04])
    assert torch.allclose(result.child.child, expected, atol=1e-01)

    # Test for Taylor series
    # Taylor series approximations for exp give error. Needs correction.


def test_exp_fixprecision():

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).fix_precision().poly()
    result = x.exp()
    expected = torch.tensor([11302, 19502, 22736, 57406, 207530, 598502, 1362052, 2562326])
    assert torch.allclose(result.child.child.child, expected, atol=1e-01)
    # Do the same with Taylor series


def test_sigmoid_fixprecision():

    x = (
        torch.tensor([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        .fix_precision()
        .poly(method="interpolation")
    )
    result = x.sigmoid()
    expected = torch.tensor([134, 205, 293, 394, 500, 606, 707, 795, 866, 929, 848])
    assert torch.allclose(result.child.child.child, expected, atol=1e01)

    # Do the same with Taylor series


def test_tanh_fixprecision():
    hook = sy.TorchHook(torch)
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
        .get()
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

    # assert torch.allclose(result.child.child, expected, atol=1e01)

    # Do the same with Taylor series


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

    """print(torch.tensor[1.1302e01, 4.1313e01, 2.9646e03, 2.1985e04].fix_precision())
    x = torch.tensor([1, 4.0, 8.0,10.0]).fix_precision().poly()
    result = x.exp()
    print(result)
    expected = torch.tensor([1.1302e01, 4.1313e01, 2.9646e03, 2.1985e04])
    #assert torch.allclose(result.child, expected, atol=1e-01)"""


"""
def test_tanh():
    # Test if tanh approximation works as expected

    poly_tensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)

    expected = torch.tensor(
        [-0.9937, -0.9811, -0.9329, -0.7596, -0.3239, 0.3239, 0.7596, 0.9329, 0.9811, 0.9937],
        dtype=torch.double,
    )

    result = poly_tensor.get_val("tanh", x)

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, result, atol=1e-03)


def test_interpolate():

    # Test if interpolation function works as expected by verifying an approximation of exponential function

    expected = torch.tensor([1.2220, 2.9582, 7.1763, 20.3064, 54.4606], dtype=torch.double)

    poly_tensor = PolynomialTensor()

    f1 = poly_tensor.interpolate((lambda x: np.exp(x)), np.linspace(0, 10, 50))

    assert torch.allclose(expected, torch.tensor(f1(torch.tensor([0, 1, 2, 3, 4]))), 1e-04)


def test_custom_function():
    poly_tensor = PolynomialTensor()
    poly_tensor.add_function(
        "Custom", 10, [[0, 10, 100, 10, poly_tensor.fit_function]], lambda x: x + 2
    )

    assert round(poly_tensor.get_val("Custom", 3)) == 5
    assert round(poly_tensor.get_val("Custom", 6)) == 8


def test_random_function():

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)
    poly_tensor = PolynomialTensor(function=lambda x: x * 2)
    scaled_result = poly_tensor.get_val("exp", x.clone())

    poly_tensor = PolynomialTensor()
    original_result = poly_tensor.get_val("exp", x)

    assert torch.all(torch.eq(scaled_result, torch.mul(original_result, 2)))


def test_log_function():

    # Test if log approximation works as expected
    poly_tensor = PolynomialTensor()
    x = torch.tensor(np.linspace(1, 10, 10), dtype=torch.double)
    expected = torch.tensor(
        [
            3.7160e-04,
            6.9319e-01,
            1.0986e00,
            1.3863e00,
            1.6096e00,
            1.7932e00,
            1.9526e00,
            2.1056e00,
            2.2835e00,
            2.5537e00,
        ],
        dtype=torch.double,
    )

    result = poly_tensor.get_val("log", x.clone())

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, result, atol=1e-03

def test_exp_taylor():
    expected = torch.tensor(
        [-0.1076, 0.0664, 0.1852, 0.3677, 0.7165, 1.3956, 2.7180, 5.2867, 10.2325, 19.5933],
        dtype=torch.double,
    )
    poly_tensor = PolynomialTensor()
    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)
    result = poly_tensor.exp(x)
    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, result, atol=1e-03)


def test_sigmoid_taylor():

    expected = torch.tensor(
        [0.1000, 0.1706, 0.2473, 0.3392, 0.4447, 0.5553, 0.6608, 0.7527, 0.8294, 0.9000],
        dtype=torch.double,
    )
    poly_tensor = PolynomialTensor()
    x = torch.tensor(np.linspace(-2, 2, 10), dtype=torch.double)
    result = poly_tensor.sigmoid(x)
    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, result, atol=1e-03) """
