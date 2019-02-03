
from syft.frameworks.torch.tensors.Polynomial import PolynomialTensor

import math

""" Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error between actual funcion values and approximations do not deviate too much"""

tensor = PolynomialTensor()
# Maximum permissible error as calculated by EvalRelative under PolynomialTensor
threshold = 0.1

def EvalRelative(x_true, x_pred):

        """The function is used to measure the error between actual function value and approximated function value. The error is evaluated with respect to actual value.
           
        
            Parameters: 
            
            x_true: Value of true function 
            x_pred: Value of function approximation 
            
        """

        error = abs(x_true - x_pred)
        return round((error / x_true), 2)

def EvalRelative_Test(x_true, x_pred):

    assert EvalRelative(15 / 7.5) == 0.5
    assert EvalRelative(15 / 1.5) == 0.9
    assert EvalRelative(15 / 15) == 0

def SigmoidTest(self):

    test_range = 4
    for i in range(-test_range, test_range, 1):

         assert (
            tensor.EvalRelative(tensor.exp(i / 10), math.exp(i / 10))
            < threshold
         )

def ExpTest():

    test_range = 10
    for i in range(-test_range, test_range, 1):

            assert (
                   tensor.EvalRelative(tensor.exp(i / 10), math.exp(i / 10))
                < threshold
            )

def tanhTest():

    test_range = 9
    for i in range(-test_range, test_range, 1):

            assert (
                 tensor.EvalRelative(tensor.tanh(i / 10), math.tanh(i / 10))< threshold
            )

def LogTest():

    test_range = 4
    for i in range(1, test_range, 1):

            assert (
                   EvalRelative(tensor.log(i / 10), math.log(i / 10))<threshold
            )
