from syft.frameworks.torch.tensors.interpreters.Polynomial import PolynomialTensor
import torch


""" Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error implementation of function approximations work as expected"""

# Maximum permissible error as calculated by EvalError under PolynomialTensor

MARGIN = 0.01

LOWER_RANGE = -3
HIGHER_RANGE = 3

STEPS = 10


def EvalError(x_true, x_pred):

    """ The function is used to find out if the actual value and approximated value are the same within a margin of error
        given by MARGIN. Since there would be changes in approximation formula leading to small differences and differences
        due to floating points. We want to ensure that the value of actual function value and approximated function value
        differ by only small margin defined by RANGE.
        
        Arguments:
            
            x_true (Torch Tensor): True value
            x_pred (Torch Tensor): Expected value
            
        Returns:
            
            within range (boolean): Indicates if the values are within margin of error
            
        """

    within_range = torch.abs(x_true - x_pred) < MARGIN
    within_range = torch.eq(within_range.all(), 1)
    within_range = within_range.item() == 1

    return within_range


def test_EvalError():

    " Ensure that EvalError function implementation works the way it has to"

    ones = torch.ones([5])
    zero = torch.zeros([5])

    margin1 = ones + (MARGIN - 0.002)
    margin2 = ones + (MARGIN + 0.001)

    assert EvalError(ones, zero) == False
    assert EvalError(ones, ones) == True
    assert EvalError(ones, margin1) == True
    assert EvalError(ones, margin2) == False


def testSigmoid():

    Ptensor = PolynomialTensor()

    x = torch.linspace(-3, 3, steps=10)
    expected = torch.tensor(
        [-0.1938, 0.0372, 0.1530, 0.2688, 0.4174, 0.5826, 0.7313, 0.8470, 0.9628, 1.1938]
    )

    assert EvalError(expected, Ptensor.sigmoid(x)) == True


def testExp():

    Ptensor = PolynomialTensor()

    x = torch.linspace(-3, 3, steps=10)
    expected = torch.tensor(
        [0.2179, 0.1224, 0.1905, 0.3679, 0.7165, 1.3956, 2.7179, 5.2813, 10.1764, 19.2679]
    )

    assert EvalError(expected, Ptensor.exp(x)) == True


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

    assert EvalError(expected, Ptensor.tanh(x)) == True
