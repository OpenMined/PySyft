

class PolynomialTensor():

    """Tensor type which provides polynomial approximation functionalities using Taylor Series expansion
       since computing exact functions could impose a overhead on computation
    """

    def sigmoid(self, x: float) -> float:

        """Parameters:
            
           x: A float value
           
           return: 
               
           approximation of the sigmoid function as a float"""

        return (1 / 2) + (x / 4) - (x ** 3 / 48) + (x ** 5 / 480)

    def exp(self, x: float, order: int) -> float:

        """Parameters:
            
           x: (float) value for which approximation must be found
           order: (int) order of Taylor series expansion

           return: 
               
           value: (float) approximation of the exponential function """

        denom = 1
        value = 1

        for i in range(1, order):

            value += (x ** order) / denom
            denom *= i + 1

        return value

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

    def EvalRelative(x_true, x_pred):

        """The function is used to measure the error between actual function value and approximated function value. The error is evaluated with respect to actual value.
           
        
            Parameters: 
            
            x_true: Value of true function 
            x_pred: Value of function approximation 
            
        """

        error = abs(x_true - x_pred)
        return round((error / x_true), 2)
