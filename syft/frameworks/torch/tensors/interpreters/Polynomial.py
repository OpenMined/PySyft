from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor


class PolynomialTensor(AbstractTensor):

    """Tensor type which provides polynomial approximation functionalities using Taylor Series expansion
       since computing exact functions could impose a overhead on computation
    """

    def sigmoid(self, x: float) -> float:

        """Parameters:
            
           x: A float value
           
           return: 
               
           approximation of the sigmoid function as a float"""

        return (1 / 2) + (x / 4) - (x ** 3 / 48) + (x ** 5 / 480)

    def exp(self, x: float) -> float:

        """Parameters:
            
           x: (float) value for which approximation must be found
           
           return: 
               
           value: (float) approximation of the exponential function """

        return 1 + x + (x ** 2 / 2) + (x ** 3 / 6) + (x ** 4 / 24)

    def tanh(self, x: float) -> float:

        """Parameters:
            
            x: A float value
           
            return: 
               
            approximation of the tanh function as a float"""

        return (x) + ((x ** 3) / 3) + (((x ** 5) / 120)) + ((x ** 7) / 5040)

    def log(self, x: float) -> float:

        """Parameters:
            
            x: A float value
           
            return: 
               
            approximation of the log function as a float"""

        return (x) - ((x ** 2) / 2) + ((x ** 3) / 3) - ((x ** 4) / 4)
