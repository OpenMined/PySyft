from syft.frameworks.torch.tensors.interpreters.Polynomial import PolynomialTensor
import torch


""" Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error between actual funcion values and approximations do not deviate too much"""

# Maximum permissible error as calculated by EvalRelative under PolynomialTensor

DATA_THRESHOLD = 2.9
ERROR_THRESHOLD = 0.05


def EvalError(x_true, x_pred):

    """ The function is used to measure the error between actual function value and approximated function value. It is evaluated as follows.
        Given a tensor of 1000 random numbers, the function approximation of individual numbers in the tensor must be within atleast +- 5% error of actual value.
        This threshold is given by ERROR_THRESHOLD.
        But , a certain percentage of numbers in the tensor can exceed the ERROR_THRESHOLD. This threshold is described by DATA_THRESHOLD.
        The function returns true if the percentage of individual approximations are within the DATA_THRESHOLD.
        
            Parameters: 
            
            x_true: Value of true function 
            x_pred: Value of function approximation 
            
            Returns:
                
            within_limits: If function approximation error of given tensor is within DATA_THRESHOLD
            
        """
    
    #Relative absolute error give by abs(true_value-prediction_value)/(true_value)
    error_rel = torch.div(torch.abs(x_true - x_pred), x_true)
    
    #Check if individual approximations are within the threshold
    error = error_rel < ERROR_THRESHOLD

    count = 0

    for i in error:

        if i.item() == 0:
            count += 1

    data_error = (count / len(x_true)) * 100
    print(data_error)
    within_limits = data_error < DATA_THRESHOLD

    return within_limits


def test_EvalRelative():

    " Ensure that EvalRelative function does what it has to. For the purpose we take a tensor of ones and subtract and add an amount of error "

    ones = torch.ones((2, 3))
    pred = ones - 0.1
    assert EvalError(ones, pred) == 0.1

    ones = torch.ones((2, 3))
    pred = ones - 0.2
    assert EvalError(ones, pred) == 0.2

    ones = torch.ones((2, 3))
    pred = ones + 0.2
    assert EvalError(ones, pred) == 0.2

    ones = torch.ones((2, 3))
    pred = ones + 0.1
    assert EvalError(ones, pred) == 0.1

    ones = torch.ones((2, 3)) * 2
    pred = ones + 0.2
    assert EvalError(ones, pred) == 0.1

    ones = torch.ones((2, 3)) * 2
    pred = ones - 0.2
    assert EvalError(ones, pred) == 0.1


def testSigmoid():

    Ptensor = PolynomialTensor()

    x = torch.randn(50000) 
    
    print(x)

    m = torch.nn.Sigmoid()
    ten = m(x)

    assert (EvalError(ten, Ptensor.sigmoid(x))) == True


def testExp():

    Ptensor = PolynomialTensor()

    x = torch.randn(50000) 
    
    print(x)

    ten = torch.exp(x)

    assert (EvalError(ten, Ptensor.exp(x))) == True


def testtanh():

    Ptensor = PolynomialTensor()

    x = torch.randn(50000) 
    
    print(x)

    m = torch.nn.tanh()
    ten = m(x)

    assert (EvalError(ten, Ptensor.tanh(x))) == True


def testLog():

    Ptensor = PolynomialTensor()

    x = torch.randn(50000) 
    
    print(x)

    m = torch.nn.tanh()
    ten = m(x)

    assert (EvalError(ten, Ptensor.log(x))) == True
    
testExp()

