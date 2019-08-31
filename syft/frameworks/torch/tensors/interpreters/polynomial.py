from syft.generic.tensor import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded
from syft.frameworks.torch.tensors.interpreters import FixedPrecisionTensor
import torch
import numpy as np
from typing import Callable, List, Union


class PolynomialTensor(AbstractTensor):
    """
    Tensor type to provide non-linear function approximations

    MPC and Homomorphic Encryption are capable of performing some addition and logical operations.
    Non-linear functions could be approximated as a series of approximated functions of basic arithmetic
    operations using function approximations such as interpolation/Taylor series.

    The polynomial tensor provides flexibility to consider every non-linear function as piecewise linear function
    and fit over different intervals.
    
    Args:
        function:
        precision:
    """

    def __init__(
        self,
        method="interpolation",
        precision=10,
        child=None,
        owner=None,
        id=None,
        tags: set = None,
        description: str = None,
    ):
        """
        Args:
            function[callable,Optional]: Function to applied to function approximation coefficients.
                Used to encrypt coefficients.
            precision[integer]: Precision of approximated values
        """

        super().__init__(owner=owner, id=id, tags=tags, description=description)

        self.method = method
        self.precision = precision
        # Stores parameters of function approximations such as precision, degree, piecewise functions and base function
        self.function_attr = {}
        # Stores fitted function
        self.func_approx = {}
        self.child = child
        self.id = None
        self.encrypt_fn = {"exp": None, "sigmoid": None}

        def default_functions():
            """Initializes default function approximations exp, log, sigmoid and tanh"""

            self.add_function(
                "exp", lambda x: np.exp(x), degree=10, min_val=-10, max_val=10, steps=100
            )
            # self.add_function("log", 10,-10,10,10, lambda x: np.log(x))
            self.add_function(
                "sigmoid",
                (lambda x: 1 / (1 + np.exp(-x))),
                degree=10,
                min_val=-10,
                max_val=10,
                steps=100,
            )
            self.add_function(
                "tanh", lambda x: np.tanh(x), degree=10, min_val=-10, max_val=10, steps=100
            )

        default_functions()

    def get_encrypt_function(self, coeffs_tensor):
        """The method encrypts the coefficients as required by the child tensor"""

        tensor = None

        if isinstance(self.child, FixedPrecisionTensor):

            tensor = getattr(torch.tensor(coeffs_tensor), "fix_precision")()

        else:

            tensor = torch.tensor(coeffs_tensor)

        return tensor

    def add_function(self, name, function, degree=10, min_val=-10, max_val=10, steps=100):
        """Add function to function_attr dictionary.

           Args:
             name[str]: Name of function
             degree[int]: Degree of function
             piecewise[List]: List of piecewise functions in format [min_val of fit,max_val of fit,step of fit,function to fit values]
             function[callable]: Base function
           """

        self.function_attr[name + "_degree"] = degree
        self.function_attr[name + "_function"] = function
        self.func_approx[name] = self.interpolate(
            function, [min_val, max_val, steps], degree=degree
        )

    def interpolate(
        self, function: Callable, interval: List[Union[int, float]], degree: int = 10
    ) -> np.poly1d:

        """Returns a interpolated version of given function using Numpy's polyfit method.
        
           Ref: https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/

         Args:
             function (a lambda function): Base function to be approximated
             interval (list of floats/integers): Interval of values to be approximated
             degree (Integer): Degree of polynomial approximation
             precision (Integer): Precision of coefficients

         Returns:
             f_interpolated (Numpy poly1d): Approximated Function
         """

        # function we wish to approximate
        f_real = function
        # interval over which we wish to optimize
        f_interval = np.linspace(interval[0], interval[1], interval[2])

        # interpolate polynomial of given max degree
        coefs = np.polyfit(f_interval, f_real(f_interval), degree)

        # reduce precision of interpolated coefficients
        precision = self.precision
        coefs = [int(x * 10 ** precision) / 10 ** precision for x in coefs]

        # approximation function
        f_interpolated = np.poly1d(coefs)

        return f_interpolated

    def sigmoid(self, method="interpolation", degree=10, precision=10):
        """Method provides Sigmoid function approximation interms of Taylor Series.
           Range:
               
               Taylor Series:[?,?]
               Interpolation:[-10.0,10.0] (Default , can be fitted to other ranges)

         Args:
             x: Torch tensor

         Returns:
             approximation of the sigmoid function as a torch tensor
         """

        sigmoid_coeffs = None
        if self.method == "taylor":

            sigmoid_coeffs = torch.tensor(
                [(1 / 2), (1 / 4), 0, -(1 / 48), 0, (1 / 480), 0, -(17 / 80640), 0, (31 / 1451520)]
            )

        elif self.method == "interpolation":

            sigmoid_coeffs = torch.tensor(self.func_approx["sigmoid"].coef)

        else:

            raise ValueError(self.method + " method of evaluating polynomials not found ")

        sigmoid_coeffs = self.get_encrypt_function(sigmoid_coeffs)
        val = 0
        x = self.child

        if hasattr(sigmoid_coeffs, "child"):

            for i in range(0, len(sigmoid_coeffs)):

                val += (x ** i) * sigmoid_coeffs[(len(sigmoid_coeffs) - 1) - i].child

        else:

            for i in range(0, len(sigmoid_coeffs)):

                val += (x ** i) * sigmoid_coeffs[(len(sigmoid_coeffs) - 1) - i]

        Ptensor = PolynomialTensor()
        Ptensor.child = val
        return Ptensor

    __sigmoid__ = sigmoid

    def tanh(self, method="interpolation", degree=10, precision=10):
        """Method provides tanh function approximation interms of Taylor Series.
           Range:
               
               Taylor Series:[?,?]
               Interpolation:[-10.0,10.0] (Default , can be fitted to other ranges)

         Args:
             x: Torch tensor

         Returns:
             approximation of the sigmoid function as a torch tensor
         """

        tanh_coeffs = None
        if self.method == "taylor":

            tanh_coeffs = torch.tensor([(1), 0, (-1 / 3), 0, (2 / 15), 0, (-17 / 315)])

        elif self.method == "interpolation":

            tanh_coeffs = torch.tensor(self.func_approx["tanh"].coef)

        else:

            raise ValueError(self.method + " method of evaluating polynomials not found ")

        tanh_coeffs = self.get_encrypt_function(tanh_coeffs)
        val = 0
        x = self.child
        if hasattr(tanh_coeffs, "child"):

            for i in range(0, len(tanh_coeffs)):

                val += (x ** i) * tanh_coeffs[(len(tanh_coeffs) - 1) - i].child

        else:

            for i in range(0, len(tanh_coeffs)):

                val += (x ** i) * tanh_coeffs[(len(tanh_coeffs) - 1) - i]

        Ptensor = PolynomialTensor()
        Ptensor.child = val
        return Ptensor

    __tanh__ = tanh

    def exp(self, method="interpolation", degree=10, precision=10):

        exp_coeffs = None
        if self.method == "taylor":

            exp_coeffs = torch.tensor(
                [1, (1 / 2), (1 / 6), (1 / 24), (1 / 120), (1 / 840), (1 / 6720)]
            )

        elif self.method == "interpolation":

            exp_coeffs = torch.tensor(self.func_approx["exp"].coef)

        else:

            raise ValueError(self.method + " method of evaluating polynomials not found ")

        exp_coeffs = self.get_encrypt_function(exp_coeffs)
        val = 0
        x = self.child
        if hasattr(exp_coeffs, "child"):

            for i in range(0, len(exp_coeffs)):

                val += (x ** i) * exp_coeffs[(len(exp_coeffs) - 1) - i].child

        else:

            for i in range(0, len(exp_coeffs)):

                val += (x ** i) * exp_coeffs[(len(exp_coeffs) - 1) - i]

        Ptensor = PolynomialTensor()
        Ptensor.child = val
        return Ptensor

    __exp__ = exp

    """def piecewise_linear_eval(self, data, x):
        Get approximated value for a given function. This takes only scalar value.
         If you have a Numpy array or torch tensor consider passing it using a lambda
         or torch.apply_ method.

         Args:
             data[2D List]: Instance of piecewise linear fit taking values [min_val, max_val,
                 function approximation method]
             x[Float or Integer]: Value to be approximated
         

        for element in data:

            min_val = element[0]
            max_val = element[1]
            function = element[2]

            if min_val <= x <= max_val:

                return function(x)
    """

    """@staticmethod
    @overloaded.module
    def torch(module):
        def exp(self, x):
            return self.__exp__(x)

        # Just register it using the module variable
        module.exp = exp

        @overloaded.function
        def sigmoid(self, x):
            return self.__sigmoid__(x)

        module.sigmoid = sigmoid

        @overloaded.function
        def tanh(self, x):
            return self.__tanh__(x)

        # Just register it using the module variable
        module.tanh = tanh"""

    """    def piecewise_linear_fit(self, name, array):
    Fit a piecewise linear function. This can be used to approximate a non-linear function
         as separate linear functions valid for separate ranges.
         For instance function approximations are more accurate for exponential when separate instances
         of interpolation are fit between -10 to 0 and 0 to 10.

         Args:
             array[2D List]: Each instance of list must take four values [min_val, steps, max_val,
                 function approximation method]

         Returns:
             array[2D List]: Each instance of list with four values [min_val,max_val,Approximated function]
         

        arguments = []

        for element in array:

            min_val = element[0]
            max_val = element[1]
            steps = element[2]
            degree = element[3]
            function = element[4]
            arguments.append(
                [
                    min_val,
                    max_val,
                    function(name, min_val=min_val, max_val=max_val, steps=steps, degree=degree),
                ]
            )

        return arguments"""

    """def fit_function(self, name, min_val=0, max_val=10, steps=100, degree=10) -> np.poly1d:
        Interpolated approximation of given function"

         Args:
             name: Name of function as defined in self.setting
             min_val: Minimum range of interpolation fit
             max_val: Maximum range of interpolation fit
             steps:   Steps of interpolation fit
             degree: Degree of interpolation fit
             function: The function used to encrypt function approximation coefficients

         Returns:
             f_interpolated (Numpy Poly1d): Approximated function
         

        fitted_function = self.interpolate(
            self.function_attr[name + "_" + "function"],
            np.linspace(min_val, max_val, steps),
            degree=degree,
        )

        fitted_function = self.apply_coefs(fitted_function, self.function)

        return np.poly1d(fitted_function)"""
