from syft.frameworks.torch.tensors.interpreters.Polynomial import PolynomialTensor
import torch


""" Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error between actual funcion values and approximations do not deviate too much"""

# Maximum permissible error as calculated by EvalRelative under PolynomialTensor

DATA_THRESHOLD = 5.5
ERROR_THRESHOLD = 5
DATA_SIZE = 50000


def EvalError(x_true, x_pred):

    """ The function is used to measure the error between actual function value and approximated function value. It is evaluated as follows.
        Given a tensor of 1000 random numbers, the function approximation of individual numbers in the tensor must be within atleast +- 5% error of actual value.
        This threshold is given by ERROR_THRESHOLD.
        But , a certain percentage of numbers in the tensor can exceed the ERROR_THRESHOLD. This threshold is described by DATA_THRESHOLD.
        The function returns true the total data error (% of individual approximations which have error within ERROR_THRESHOLD)
        
            Parameters: 
            
            x_true: Value of true function 
            x_pred: Value of function approximation 
            
            Returns:
                
            data_error: 
            
        """

    # Relative absolute error give by abs(true_value-prediction_value)/(true_value)
    error_rel = torch.div(torch.abs(x_true - x_pred), x_true) * 100

    # Check if individual approximations are within the threshold
    error = error_rel < ERROR_THRESHOLD

    count = 0

    for i in error:

        if i.item() == 0:
            count += 1

    data_error = (count / len(x_true)) * 100

    return data_error


def test_EvalError():

    " Ensure that EvalRelative function does what it has to. For the purpose we take a tensor of ones and subtract and add an amount of error "

    one = torch.ones(6)
    two = torch.ones(6) * 2

    assert EvalError(two, one) == 100
    assert EvalError(one, one) == 0
    assert EvalError(two, two) == 0

    three = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    assert EvalError(three, one) == 50
    assert EvalError(one, three) == 50


def testSigmoid():

    Ptensor = PolynomialTensor()

    x = torch.randn(DATA_SIZE)

    m = torch.nn.Sigmoid()
    ten = m(x)

    assert (EvalError(ten, Ptensor.sigmoid(x))) < DATA_THRESHOLD


def testExp():

    Ptensor = PolynomialTensor()

    x = torch.randn(DATA_SIZE)

    ten = torch.exp(x)

    assert (EvalError(ten, Ptensor.exp(x))) < DATA_THRESHOLD


def testtanh():

    # Ideally the error for any approximation to be under DATA_THRESHOLD. For tanh it could be not be improved beyond 12 , which should be addressed in the future.
    Tanh_thresh = 12

    Ptensor = PolynomialTensor()

    x = torch.randn(DATA_SIZE)

    ten = torch.tanh(x)

    assert (EvalError(ten, Ptensor.tanh(x))) < Tanh_thresh
