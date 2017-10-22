# coding=utf-8
"""
    Module activations provides activation layers for neural networks.
    Note:The Documentation in this file follows the NumPy Doc. Style;
         Hence, it is mandatory that future docs added here
         strictly follow the same, to maintain readability and consistency
         of the codebase.
    NumPy Documentation Style-
        http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
"""

import numpy as np
from syft.nonlin import PolyFunction
from syft.encryptable import Encryptable

def SquareActivation():
    """
    Returns
    -------
    Activation:
        x^2 activation function.
    """
    square = PolyFunction([1,0,0])
    return PolynomialActivation(square)

def LinearSquareActivation():
    """
    Returns
    -------
    Activation:
        x^2+x activation function.
    """
    linear_square = PolyFunction([1,1,0])
    return PolynomialActivation(linear_square)

def CubicActivation():
    """
    Returns
    -------
    Activation:
        x^3 activation function.
    """
    cubic = PolyFunction([1,0,0,0])
    return PolynomialActivation(cubic)

def LinearCubicActivation():
    """
    Returns
    -------
    Activation:
        x^3+x activation function.
    """
    linear_cubic = PolyFunction([1,0,1,0])
    return PolynomialActivation(linear_cubic)

def SigmoidActivation():
    """
    Returns
    -------
    Activation:
        Sigmoid activation function.
    """
    sigmoid = PolyFunction([0.0000000072,0,-0.0000018848,0,0.0001825597,0,-0.0082176259,0,0.2159198015,0.5])
    sigmoid.derivative = np.vectorize(lambda x: (1 - x) * x)
    return PolynomialActivation(sigmoid)

class Activation:
    """
    Generic neural network activation layer.
    """
    def __init__(self):
        pass
        
    def __call__(self,x):
        pass
        
    def forward(self, x):
        pass
    
    def backward(self, x):
        pass
    
    
class PolynomialActivation(Activation, Encryptable):
    """
    Polynomial-based activation layer.
    """
    encryptables=['polynomial']
    def __init__(self, polynomial):
        self.polynomial = polynomial
        super().__init__()
        
    def __call__(self,x):
        """
        Evaluates the activation on the specified input.
        Parameters
        ----------
        x : numpy array
            Values to evaluate the activation on.
        Returns
        -------
        numpy array:
            Polynomial activation values in x.
        """
        return self.polynomial(x)
        
    def forward(self, x):
        """
        Evaluates the activation on the specified input.
        Parameters
        ----------
        x : numpy array
            Values to evaluate the activation on.
        Returns
        -------
        numpy array:
            Polynomial activation values in x.
        """
        return self.polynomial(x)
    
    def backward(self, x):
        """
        Evaluates the derivative of the specified input.
        Parameters
        ----------
        x : numpy array
            Values to evaluate the derivative activation on.
        Returns
        -------
        numpy array:
            Polynomial derivative values in x.
        """
        return self.polynomial.derivative(x)
    
