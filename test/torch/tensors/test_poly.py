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
        [0.2179, 0.1224, 0.1905, 0.3679, 0.7165, 1.3956, 2.7179, 5.2813, 10.1764, 19.2679],
        dtype=torch.double,
    )

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(Ptensor.exp(x), dtype=torch.double), 1e-03)


def testtanh():

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
