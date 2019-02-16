from syft.frameworks.torch.tensors.interpreters.Polynomial import PolynomialTensor
import torch


""" Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error between actual funcion values and approximations do not deviate too much"""

# Maximum permissible error as calculated by EvalRelative under PolynomialTensor

DATA_THRESHOLD=3
ERROR_THRESHOLD =0.15


def EvalRelative(x_true, x_pred):

    """The function is used to measure the error between actual function value and approximated function value. The error is evaluated with respect to actual value.
           
        
            Parameters: 
            
            x_true: Value of true function 
            x_pred: Value of function approximation 
            
        """

    error = torch.div(torch.abs(x_true - x_pred), x_true)
    
    error=error<ERROR_THRESHOLD
        
    count=0
    
    for i in error:
        
        if(i.item()==0):
            
            count+=1
    
    data_error=(count/len(x_true))*100
    print(data_error<DATA_THRESHOLD)

    return data_error


def test_EvalRelative():

    " Ensure that EvalRelative function does what it has to. For the purpose we take a tensor of ones and subtract and add an amount of error "

    ones = torch.ones((2, 3))
    pred = ones - 0.1
    assert EvalRelative(ones, pred) == 0.1

    ones = torch.ones((2, 3))
    pred = ones - 0.2
    assert EvalRelative(ones, pred) == 0.2

    ones = torch.ones((2, 3))
    pred = ones + 0.2
    assert EvalRelative(ones, pred) == 0.2

    ones = torch.ones((2, 3))
    pred = ones + 0.1
    assert EvalRelative(ones, pred) == 0.1
    
    ones = torch.ones((2, 3))*2
    pred = ones + 0.2
    assert EvalRelative(ones, pred) == 0.1
    
    ones = torch.ones((2, 3))*2
    pred = ones - 0.2
    assert EvalRelative(ones, pred) == 0.1


def testSigmoid():

    Ptensor = PolynomialTensor()

    x = torch.randn(100000)*(-1)
    
    
    m = torch.nn.Sigmoid()
    ten = m(x)


    assert (EvalRelative(ten, Ptensor.sigmoid(x))) < threshold


def testExpTest():

    Ptensor = PolynomialTensor()
    in_tensor = torch.randn(2)

    m = torch.nn.Sigmoid()
    ten = m(in_tensor)

    assert EvalRelative(ten, Ptensor.sigmoid(in_tensor)) < threshold


def testtanh():

    Ptensor = PolynomialTensor()
    in_tensor = torch.randn(2)

    m = torch.nn.Sigmoid()
    ten = m(in_tensor)

    assert EvalRelative(ten, Ptensor.sigmoid(in_tensor)).all() < threshold


def testLog():

    Ptensor = PolynomialTensor()
    in_tensor = torch.randn(2)

    m = torch.nn.Sigmoid()
    ten = m(in_tensor)

    assert EvalRelative(ten, Ptensor.sigmoid(in_tensor)).all() < threshold


testSigmoid()
