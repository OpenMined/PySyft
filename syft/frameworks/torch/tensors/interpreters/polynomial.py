from syft.generic.tensor import AbstractTensor
from syft.generic.frameworks.overload import overloaded
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
import torch
import syft
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

        def default_functions():

            """Initializes default function approximations exp, log, sigmoid and tanh"""

            self.add_function(
                "exp", lambda x: np.exp(x), degree=10, min_val=-10, max_val=10, steps=100
            )
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

    @property
    def grad(self):
        """
        Gradient makes no sense for Polynomial Tensor, so we make it clear
        that if someone query .grad on a Fixed Precision Tensor it doesn't error
        but returns grad and can't be set
        """
        return None

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

        self.function_attr[name] = [degree, min_val, max_val, steps]
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

    @overloaded.method
    def sigmoid(
        self, method="interpolation", degree=10, precision=10, min_val=-10, max_val=10, steps=100
    ):
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

            # Check if the user wants a fitted approximation different from already fitted function and fit accordingly
            if self.function_attr["sigmoid"] == [degree, min_val, max_val, steps]:

                sigmoid_coeffs = torch.tensor(self.func_approx["sigmoid"].coef)

            else:

                self.add_function(
                    "sigmoid",
                    (lambda x: 1 / (1 + np.exp(-x))),
                    degree=degree,
                    min_val=min_val,
                    max_val=max_val,
                    steps=steps,
                )
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

        return val

    __sigmoid__ = sigmoid

    @overloaded.method
    def tanh(
        self, method="interpolation", degree=10, min_val=-10, max_val=10, steps=100, precision=10
    ):
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

            # 2 * sigmoid(2 * tensor) - 1
            tanh_coeffs = torch.tensor(
                [
                    0,
                    (1),
                    0,
                    (-1 / 3),
                    0,
                    (2 / 15),
                    0,
                    (-17 / 315),
                    0,
                    (62 / 2835),
                    0,
                    (-1382 / 155925),
                ]
            )

        elif self.method == "interpolation":

            # Check if the user wants a fitted approximation different from already fitted function and fit accordingly
            if self.function_attr["tanh"] == [degree, min_val, max_val, steps]:

                tanh_coeffs = torch.tensor(self.func_approx["tanh"].coef)

            else:

                self.add_function(
                    "tanh",
                    (lambda x: np.tanh(x)),
                    degree=degree,
                    min_val=min_val,
                    max_val=max_val,
                    steps=steps,
                )
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

        return val

    __tanh__ = tanh

    @overloaded.method
    def exp(
        self, method="interpolation", degree=10, min_val=-10, max_val=10, steps=100, precision=10
    ):

        exp_coeffs = None
        if self.method == "taylor":

            exp_coeffs = torch.tensor([1, (1 / 2), (1 / 6), (1 / 24), (1 / 120)])

        elif self.method == "interpolation":

            # Check if the user wants a fitted approximation different from already fitted function and fit accordingly
            if self.function_attr["exp"] == [degree, min_val, max_val, steps]:

                exp_coeffs = torch.tensor(self.func_approx["exp"].coef)

            else:

                self.add_function(
                    "tanh",
                    (lambda x: np.exp(x)),
                    degree=degree,
                    min_val=min_val,
                    max_val=max_val,
                    steps=steps,
                )
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

        return val

    __exp__ = exp

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a FixedPrecision Tensor,
        Perform some specific action (like logging) which depends of the
        instruction content, replace in the args all the FPTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a FixedPrecision on top of all tensors found in
        the response.
        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        cmd, _, args, kwargs = command

        tensor = args[0] if not isinstance(args[0], (tuple, list)) else args[0][0]

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # TODO: I can't manage the import issue, can you?
        # Replace all FixedPrecisionTensor with their child attribute
        new_args, new_kwargs, new_type = syft.frameworks.torch.hook_args.unwrap_args_from_function(
            cmd, args, kwargs
        )

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back FixedPrecisionTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            cmd, response, wrap_type=cls, wrap_args=tensor.get_class_attributes()
        )

        return response

    @staticmethod
    @overloaded.module
    def torch(module):
        """
        We use the @overloaded.module to specify we're writing here
        a function which should overload the function with the same
        name in the <torch> module
        :param module: object which stores the overloading functions
        Note that we used the @staticmethod decorator as we're in a
        class
        """

        def add(x, y):
            """
            You can write the function to overload in the most natural
            way, so this will be called whenever you call torch.add on
            Logging Tensors, and the x and y you get are also Logging
            Tensors, so compared to the @overloaded.method, you see
            that the @overloaded.module does not hook the arguments.
            """
            P = PolynomialTensor()
            P.child = x.child + y.child
            return P

        # Just register it using the module variable
        module.add = add

        @overloaded.function
        def mul(x, y):
            """
            You can also add the @overloaded.function decorator to also
            hook arguments, ie all the LoggingTensor are replaced with
            their child attribute
            """
            P = PolynomialTensor()
            P.child = x * y
            return P

        # Just register it using the module variable
        module.mul = mul

        # You can also overload functions in submodules!
        @overloaded.module
        def nn(module):
            """
            The syntax is the same, so @overloaded.module handles recursion
            Note that we don't need to add the @staticmethod decorator
            """

            @overloaded.module
            def functional(module):
                def relu(x):
                    print("Log function torch.nn.functional.relu")
                    return x * (x.child > 0)

                module.relu = relu

                def sigmoid(self, x):

                    print("Log function torch.nn.functional.relu")
                    return self._sigmoid(x)

                module.sigmoid = sigmoid

            module.functional = functional

        # Modules should be registered just like functions
        module.nn = nn

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
