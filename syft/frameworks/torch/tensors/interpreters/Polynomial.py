from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
import torch
import numpy as np
from typing import Callable, List, Union


class PolynomialTensor(AbstractTensor):

    """Tensor type which provides function approximations using Taylor Series/Interpolation methods. 
       For instance , non-linear operations such as relu , exp and tanh for MPC
    """

    def interpolate(
        self,
        function: Callable,
        interval: List[Union[int, float]],
        degree: int = 10,
        precision: int = 10,
    ) -> np.poly1d:

        """Returns a interpolated version of given function using Numpy's polyfit method
           
        Args: 
            
            function (a lambda function): Base function to be approximated 
            interval (list of floats/integers): Interval of values to be approximated
            degree (Integer): Degree of polynomial approximation
            precision(Integer): Precision of coefficients
            
        returns:
            
            f_interpolated (Numpy poly1d): Approximated Function 
        
        """

        # function we wish to approximate
        f_real = function

        # interval over which we wish to optimize
        f_interval = interval

        # interpolate polynomial of given max degree
        degree = 10
        coefs = np.polyfit(f_interval, f_real(f_interval), degree)

        # reduce precision of interpolated coefficients
        precision = 10
        coefs = [int(x * 10 ** precision) / 10 ** precision for x in coefs]

        # approximation function
        f_interpolated = np.poly1d(coefs)

        return f_interpolated

    def sigmoid_inter(self) -> np.poly1d:

        """Interpolated approximation of Sigmoid function
        
        returns:
            
            f_interpolated (Numpy Poly1d): Approximated Sigmoid
        
        """

        inter_sigmoid = self.interpolate(
            (lambda x: 1 / (1 + np.exp(-x))), np.linspace(-10, 10, 100)
        )

        return inter_sigmoid

    def sigmoid(self, x, function=torch.tensor):

        """Parameters:
            
           x: Torch tensor
           function: The function used to encrypt function approximation values
           
           
           return: 
               
           approximation of the sigmoid function as a torch tensor"""

        return (
            (function(1 / 2))
            + (function(x / 4))
            - (function((x ** 3) / (48)))
            + (function((x ** 5) / (480)))
        )

    def exp(self, x, function=torch.tensor):

        """Parameters:
            
            x: Torch tensor
            function: The function used to encrypt function approximation values
           
            return: 
               
            approximation of the sigmoid function as a torch tensor"""

        return (
            function(1)
            + function(x)
            + (function((x ** 2) / 2))
            + (function((x ** 3) / 6))
            + (function((x ** 4) / (24)))
            + (function((x ** 5) / (120)))
            + (function((x ** 6) / (840)))
            + (function((x ** 7) / (6720)))
        )

    def exp_inter(self, x, function=torch.tensor):

        raise NotImplementedError

    def tanh_inter(self, x, function=torch.tensor):

        raise NotImplementedError

    def tanh(self, x: torch.tensor, function=torch.tensor) -> torch.tensor:

        """Parameters:
            
            x: Torch tensor
            function: The function used to encrypt function approximation values
           
            return: 
               
            approximation of the sigmoid function as a torch tensor"""

        return (
            (function(x))
            - (function((x ** 3) / 3))
            + ((function((2 * (x ** 5)) / 15)))
            - ((function(17 * (x ** 7))) / 315)
            + ((function(62 * (x ** 9) / 2835)))
        )

    def relu(self, x, function=torch.tensor):

        raise NotImplementedError
