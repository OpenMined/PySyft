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

        return 1 + x + (x ** 2 / 2) + (x ** 3 / 6) + (x ** 4 / 24) + (x ** 5 / 120) + (x ** 6 / 840)

    def tanh(self, x: float) -> float:

        """Parameters:
            
            x: A float value
           
            return: 
               
            approximation of the tanh function as a float"""

        return (
            (x)
            - ((x ** 3) / 3)
            + (((2 * (x ** 5)) / 15))
            - ((17 * (x ** 7)) / 315)
            + (62 * (x ** 9) / (2835))
        )


# if x < 0 then f(x) = 1 / (0.5/(1+(x^2)))
# if x > 0 then f(x) = 1 / (-0.5/(1+(x^2)))+1
