

from syft.frameworks.torch.tensors.interpreters.Polynomial import PolynomialTensor
import torch
import numpy as np


""" Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error implementation of function approximations work as expected"""


def testSigmoid():

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10),dtype=torch.double)
        
    expected = torch.tensor([0.03373682, 0.08864994 , 0.17589712 , 0.29211714 ,0.42833034 ,0.57166966,0.70788286 , 0.82410288 ,0.91135006 , 0.96626318],dtype=torch.double)
    
    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(Ptensor.sigmoid_inter()(x),dtype=torch.double), 1e-03)


def testExp():

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10),dtype=torch.double)
    expected = torch.tensor(
        [0.2179, 0.1224, 0.1905, 0.3679, 0.7165, 1.3956, 2.7179, 5.2813, 10.1764, 19.2679]
    ,dtype=torch.double)

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(Ptensor.exp(x),dtype=torch.double), 1e-03)


def testtanh():

    Ptensor = PolynomialTensor()

    x = torch.tensor(np.linspace(-3, 3, 10),dtype=torch.double)
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
        ],dtype=torch.double
    )

    # allclose function to compare the expected values and approximations with fixed precision
    assert torch.allclose(expected, torch.tensor(Ptensor.tanh(x),dtype=torch.double), 1e-03)
