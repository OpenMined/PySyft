from syft.frameworks.torch.tensors.interpreters.Polynomial import PolynomialTensor
import torch
import numpy as np


""" Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error implementation of function approximations work as expected"""


def testSigmoid():

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)

    expected = torch.tensor(
        [
            0.033_736_82,
            0.088_649_94,
            0.175_897_12,
            0.292_117_14,
            0.428_330_34,
            0.571_669_66,
            0.707_882_86,
            0.824_102_88,
            0.911_350_06,
            0.966_263_18,
        ],
        dtype=torch.double,
    )

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(
        expected, torch.tensor(Ptensor.sigmoid_inter()(x), dtype=torch.double), 1e-03
    )


def testExp():

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)
    expected = torch.tensor(
        [-0.1076, 0.0664, 0.1852, 0.3677, 0.7165, 1.3956, 2.7180, 5.2867, 10.2325, 19.5933],
        dtype=torch.double,
    )

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(Ptensor.exp(x), dtype=torch.double), 1e-03)


def testtanh():

    # Test if tanh approximation works as expected

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10), dtype=torch.double)
    expected = torch.tensor(
        [
            -3.3883e02,
            -3.1835e01,
            -2.0803e00,
            -7.6790e-01,
            -3.2151e-01,
            3.2151e-01,
            7.6790e-01,
            2.0803e00,
            3.1835e01,
            3.3883e02,
        ],
        dtype=torch.double,
    )

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(Ptensor.tanh(x), dtype=torch.double), 1e-03)


def testinterpolate():

    # Test if interpolation function works as expected by verifying an approximation of exponential function

    expected = torch.tensor([1.2220, 2.9582, 7.1763, 20.3064, 54.4606], dtype=torch.double)

    Ptensor = PolynomialTensor()

    f1 = Ptensor.interpolate((lambda x: np.exp(x)), np.linspace(0, 10, 50))

    assert torch.allclose(expected, torch.tensor(f1(torch.tensor([0, 1, 2, 3, 4]))), 1e-04)


testSigmoid()
testExp()
