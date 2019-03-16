from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
import torch
import numpy as np


class PolynomialTensor(AbstractTensor):

    """Tensor type which provides polynomial approximation functionalities using Taylor Series expansion
       since computing exact functions could impose a overhead on computation
    """

    def interpolate(self,function,interval,degree=10,precision=10):

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
        
        return(f_interpolated)
        
    def sigmoid_inter(self):
        
        return self.interpolate(lambda x: 1/(1+np.exp(-x)),np.linspace(-10, 10, 100))

    def sigmoid(self, x: torch.tensor) -> torch.tensor:

        """Parameters:
            
           x: Torch tensor
           
           return: 
               
           approximation of the sigmoid function as a torch tensor"""

        return (1 / 2) + (x / 4) - (x ** 3 / 48) + (x ** 5 / 480)

    def exp(self, x: torch.tensor) -> torch.tensor:

        """Parameters:
            
            x: Torch tensor
           
            return: 
               
            approximation of the sigmoid function as a torch tensor"""

        return 1 + x + (x ** 2 / 2) + (x ** 3 / 6) + (x ** 4 / 24) + (x ** 5 / 120) + (x ** 6 / 840)

    def tanh(self, x: torch.tensor) -> torch.tensor:

        """Parameters:
            
            x: Torch tensor
           
            return: 
               
            approximation of the sigmoid function as a torch tensor"""

        return (
            (x)
            - ((x ** 3) / 3)
            + (((2 * (x ** 5)) / 15))
            - ((17 * (x ** 7)) / 315)
            + (62 * (x ** 9) / (2835))
        )


"""
   An implementation of fast sigmoid
   if x < 0 then f(x) = 1 / (0.5/(1+(x^2)))
   if x > 0 then f(x) = 1 / (-0.5/(1+(x^2)))+1"""
