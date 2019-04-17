import syft as sy
from syft.frameworks.torch.tensors.interpreters.Polynomial import PolynomialTensor
import torch
import numpy as np

from syft.frameworks.torch.tensors.interpreters import FixedPrecisionTensor


def testSigmoid():

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)

    expected = torch.tensor(
        [0.0337, 0.0886, 0.1759, 0.2921, 0.4283, 0.5717, 0.7079, 0.8241, 0.9114, 0.9663],
        dtype=torch.double,
    )

    # allclose function to compare the expected values and approximations with fixed precision
    result = Ptensor.get_val("sigmoid", x)
    assert torch.allclose(expected, result, atol=1e-03)


def testExp():

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)

    expected = torch.tensor(
        [0.0498, 0.0970, 0.1889, 0.3679, 0.7165, 1.1176, 2.9503, 5.1088, 10.2501, 20.2955],
        dtype=torch.double,
    )

    # allclose function to compare the expected values and approximations with fixed precision
    result = Ptensor.get_val("exp", x)
    assert torch.allclose(expected, result, atol=1e-03)


def testtanh():

    # Test if tanh approximation works as expected

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)

    expected = torch.tensor(
        [-0.9937, -0.9811, -0.9329, -0.7596, -0.3239, 0.3239, 0.7596, 0.9329, 0.9811, 0.9937],
        dtype=torch.double,
    )

    result = Ptensor.get_val("tanh", x)

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(result, dtype=torch.double), atol=1e-03)


def testinterpolate():

    # Test if interpolation function works as expected by verifying an approximation of exponential function

    expected = torch.tensor([1.2220, 2.9582, 7.1763, 20.3064, 54.4606], dtype=torch.double)

    Ptensor = PolynomialTensor()

    f1 = Ptensor.interpolate((lambda x: np.exp(x)), np.linspace(0, 10, 50))

    assert torch.allclose(expected, torch.tensor(f1(torch.tensor([0, 1, 2, 3, 4]))), 1e-04)


def testcustomfunction():

    P = PolynomialTensor()
    P.addfunction("Custom", 10, [[0, 10, 100, 10, P.fit_function]], lambda x: x + 2)

    assert round(P.get_val("Custom", 3)) == 5
    assert round(P.get_val("Custom", 6)) == 8


def testrandomfunction():

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)
    P = PolynomialTensor(function=lambda x: x * 2)
    scaled_result = P.get_val("exp", x.clone())

    Ptensor = PolynomialTensor()
    original_result = Ptensor.get_val("exp", x)

    assert torch.all(torch.eq(scaled_result, torch.mul(original_result, 2)))


def testlogfunction():

    # Test if log approximation works as expected

    Ptensor = PolynomialTensor()

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

    result = Ptensor.get_val("log", x.clone())

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(result, dtype=torch.double), atol=1e-03)


def testexptaylor():

    expected = torch.tensor(
        [-0.1076, 0.0664, 0.1852, 0.3677, 0.7165, 1.3956, 2.7180, 5.2867, 10.2325, 19.5933],
        dtype=torch.double,
    )
    P = PolynomialTensor()
    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)
    result = P.exp(x)

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(result, dtype=torch.double), atol=1e-03)


def testsigmoidtaylor():

    expected = torch.tensor(
        [0.1000, 0.1706, 0.2473, 0.3392, 0.4447, 0.5553, 0.6608, 0.7527, 0.8294, 0.9000],
        dtype=torch.double,
    )
    P = PolynomialTensor()
    x = torch.tensor(np.linspace(-2, 2, 10), dtype=torch.double)
    result = P.sigmoid(x)

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(result, dtype=torch.double), atol=1e-03)
