from syft.frameworks.torch.tensors.interpreters.Polynomial import PolynomialTensor
import torch


""" Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error implementation of function approximations work as expected"""



def testSigmoid():

    Ptensor = PolynomialTensor()

    x = torch.linspace(-3, 3, steps=10)
    expected = torch.tensor(
        [-0.1938, 0.0372, 0.1530, 0.2688, 0.4174, 0.5826, 0.7313, 0.8470, 0.9628, 1.1938]
    )

    assert torch.allclose(expected, Ptensor.sigmoid(x),1e-03)


def testExp():

    Ptensor = PolynomialTensor()

    x = torch.linspace(-3, 3, steps=10)
    expected = torch.tensor(
        [0.2179, 0.1224, 0.1905, 0.3679, 0.7165, 1.3956, 2.7179, 5.2813, 10.1764, 19.2679]
    )

    assert torch.allclose(expected, Ptensor.exp(x),1e-03)


def testtanh():

    Ptensor = PolynomialTensor()

    x = torch.linspace(-3, 3, steps=10)
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
        ]
    )

    assert torch.allclose(expected, Ptensor.tanh(x),1e-03)

testSigmoid()
testExp()
testtanh()