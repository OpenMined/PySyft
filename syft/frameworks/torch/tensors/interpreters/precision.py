import torch
import warnings

import syft
from syft.frameworks.torch.nn import nn
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.abstract.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker

from syft_proto.frameworks.torch.tensors.interpreters.v1.precision_pb2 import (
    FixedPrecisionTensor as FixedPrecisionTensorPB,
)


class FixedPrecisionTensor(AbstractTensor):
    def __init__(
        self,
        owner=None,
        id=None,
        field: int = None,
        dtype: str = "long",
        base: int = 10,
        precision_fractional: int = 3,
        kappa: int = 1,
        tags: set = None,
        description: str = None,
    ):
        """Initializes a Fixed Precision tensor, which encodes all decimal point
        values using an underlying integer value.

        The FixedPrecision enables to manipulate floats over an interface which
        supports only integers, Such as _SPDZTensor.

        This is done by specifying a precision p and given a float x,
        multiply it with 10**p before rounding to an integer (hence you keep
        p decimals)

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the FixedPrecisionTensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        self.base = base
        self.precision_fractional = precision_fractional
        self.kappa = kappa
        self.dtype = dtype
        if dtype == "long":
            self.field = 2 ** 64
            self.torch_dtype = torch.int64
        elif dtype == "int":
            self.field = 2 ** 32
            self.torch_dtype = torch.int32
        else:
            # Since n mod 0 is not defined
            warnings.warn("Prefer to use dtype instead of field")
            if isinstance(field, int) and field > 0:
                if field <= 2 ** 32:
                    self.dtype = "int"
                    self.field = 2 ** 32
                    self.torch_dtype = torch.int32
                else:
                    self.dtype = "long"
                    self.field = 2 ** 64
                    self.torch_dtype = torch.int64
            else:
                # Invalid args dtype and field
                raise ValueError(
                    "Unsupported arg value for dtype. Use dtype='long' or dtype='int'."
                )

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response,
        for example precision_fractional is important when wrapping the result of a method
        on a self which is a fixed precision tensor with a non default precision_fractional.
        """
        return {
            "field": self.field,
            "base": self.base,
            "precision_fractional": self.precision_fractional,
            "kappa": self.kappa,
            "dtype": self.dtype,
        }

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_data):
        self.child = new_data.child
        return self

    @property
    def grad(self):
        """
        Gradient makes no sense for Fixed Precision Tensor, so we make it clear
        that if someone query .grad on a Fixed Precision Tensor it doesn't error
        but returns grad and can't be set
        """
        return None

    def backward(self, *args, **kwargs):
        """Calling backward on Precision Tensor doesn't make sense, but sometimes a call
        can be propagated downward the chain to an Precision Tensor (for example in
        create_grad_objects), so we just ignore the call."""
        pass

    def attr(self, attr_name):
        return self.__getattribute__(attr_name)

    def fix_precision(self, check_range=True):
        """This method encodes the .child object using fixed precision"""

        rational = self.child
        upscaled = (rational * self.base ** self.precision_fractional).long()
        if check_range:
            assert (
                upscaled.abs() < (self.field / 2)
            ).all(), (
                f"{rational} cannot be correctly embedded: choose bigger field or a lower precision"
            )

        field_element = upscaled
        field_element.owner = rational.owner
        self.child = field_element.type(self.torch_dtype)
        return self

    def float_precision(self):
        """this method returns a new tensor which has the same values as this
        one, encoded with floating point precision"""
        value = self.child.type(self.torch_dtype)
        gate = value.native_lt(0).type(self.torch_dtype)

        neg_nums = value * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums).float() / (self.base ** self.precision_fractional)

        return result

    def truncate(self, precision_fractional, check_sign=True):
        truncation = self.base ** precision_fractional

        # We need to make sure that values are truncated "towards 0"
        # i.e. for a field of 100, 70 (equivalent to -30), should be truncated
        # at 97 (equivalent to -3), not 7
        if isinstance(self.child, AdditiveSharingTensor) or not check_sign:  # Handle FPT>(wrap)>AST
            self.child = self.child / truncation
            return self
        else:
            gate = self.child.native_lt(0).type(self.torch_dtype)
            neg_nums = self.child / truncation
            pos_nums = self.child / truncation
            self.child = neg_nums * gate + pos_nums * (1 - gate)
            return self

    @overloaded.method
    def mod(self, _self, divisor):
        """
        Define the modulo operation over object instances.
        """
        if isinstance(divisor, (int, float)):
            scaled_divisor = int(divisor * self.base ** self.precision_fractional)
            if isinstance(_self, AdditiveSharingTensor):
                return getattr(_self, "mod")(scaled_divisor)
            else:
                return getattr(_self, "fmod")(scaled_divisor)

        response = getattr(_self, "fmod")(divisor)

        return response

    __mod__ = mod

    @overloaded.method
    def add(self, _self, other):
        """Add two fixed precision tensors together."""
        if isinstance(other, (int, float)):
            scaled_int = int(other * self.base ** self.precision_fractional)
            return getattr(_self, "add")(scaled_int)

        if isinstance(_self, AdditiveSharingTensor) and isinstance(other, torch.Tensor):
            # If we try to add a FPT>(wrap)>AST and a FPT>torch.tensor,
            # we want to perform AST + torch.tensor
            other = other.wrap()
        elif isinstance(other, AdditiveSharingTensor) and isinstance(_self, torch.Tensor):
            # If we try to add a FPT>torch.tensor and a FPT>(wrap)>AST,
            # we swap operators so that we do the same operation as above
            _self, other = other, _self.wrap()

        response = getattr(_self, "add")(other)

        return response

    __add__ = add
    __radd__ = add

    def add_(self, value_or_tensor, tensor=None):
        if tensor is None:
            result = self.add(value_or_tensor)
        else:
            result = self.add(value_or_tensor * tensor)

        self.child = result.child
        return self

    def __iadd__(self, other):
        """Add two fixed precision tensors together."""
        self.child = self.add(other).child

        return self

    @overloaded.method
    def sub(self, _self, other):
        """Subtracts a fixed precision tensor from another one."""
        if isinstance(other, (int, float)):
            scaled_int = int(other * self.base ** self.precision_fractional)
            return getattr(_self, "sub")(scaled_int)

        if isinstance(_self, AdditiveSharingTensor) and isinstance(other, torch.Tensor):
            # If we try to subtract a FPT>(wrap)>AST and a FPT>torch.tensor,
            # we want to perform AST - torch.tensor
            other = other.wrap()
        elif isinstance(other, AdditiveSharingTensor) and isinstance(_self, torch.Tensor):
            # If we try to subtract a FPT>torch.tensor and a FPT>(wrap)>AST,
            # we swap operators so that we do the same operation as above
            _self, other = -other, -_self.wrap()

        response = getattr(_self, "sub")(other)

        return response

    __sub__ = sub

    def __rsub__(self, other):
        return (self - other) * -1

    def sub_(self, value_or_tensor, tensor=None):
        if tensor is None:
            result = self.sub(value_or_tensor)
        else:
            result = self.sub(value_or_tensor * tensor)

        self.child = result.child
        return self

    def __isub__(self, other):
        self.child = self.sub(other).child

        return self

    @overloaded.method
    def t(self, _self, *args, **kwargs):
        """Transpose a tensor. Hooked is handled by the decorator"""
        response = getattr(_self, "t")(*args, **kwargs)

        return response

    def mul_and_div(self, other, cmd):
        """
        Hook manually mul and div to add the truncation/rescaling part
        which is inherent to these operations in the fixed precision setting
        """
        changed_sign = False
        if isinstance(other, FixedPrecisionTensor):
            assert (
                self.precision_fractional == other.precision_fractional
            ), "In mul and div, all args should have the same precision_fractional"
            assert self.base == other.base, "In mul and div, all args should have the same base"

        if isinstance(other, (int, torch.Tensor, AdditiveSharingTensor)):
            new_self = self.child
            new_other = other
        elif isinstance(other, float):
            raise NotImplementedError(
                "Can't multiply or divide a FixedPrecisionTensor with a float value"
            )

        elif isinstance(self.child, (AdditiveSharingTensor, MultiPointerTensor)) and isinstance(
            other.child, torch.Tensor
        ):
            # If operands are FPT>AST and FPT>torch.tensor,
            # we want to perform the operation on AST and torch.tensor
            if cmd == "mul":
                new_self = self.child
            elif cmd == "div":
                new_self = self.child * self.base ** self.precision_fractional
            new_other = other

        elif isinstance(other.child, (AdditiveSharingTensor, MultiPointerTensor)) and isinstance(
            self.child, torch.Tensor
        ):
            # If operands are FPT>torch.tensor and FPT>AST,
            # we swap operators so that we do the same operation as above
            if cmd == "mul":
                new_self = other.child
                new_other = self
            elif cmd == "div":
                # TODO how to divide by AST?
                raise NotImplementedError(
                    "Division of a FixedPrecisionTensor by an AdditiveSharingTensor not implemented"
                )

        elif (
            cmd == "mul"
            and isinstance(self.child, (AdditiveSharingTensor, MultiPointerTensor))
            and isinstance(other.child, (AdditiveSharingTensor, MultiPointerTensor))
        ):
            # If we try to multiply a FPT>torch.tensor with a FPT>AST,
            # we swap operators so that we do the same operation as above
            new_self, new_other, _ = hook_args.unwrap_args_from_method("mul", self, other, None)

        else:
            # Replace all syft tensor with their child attribute
            new_self, new_other, _ = hook_args.unwrap_args_from_method(cmd, self, other, None)

            # To avoid problems with negative numbers
            # we take absolute value of the operands
            # The problems could be 1) bad truncation for multiplication
            # 2) overflow when scaling self in division

            # sgn_self is 1 when new_self is positive else it's 0
            # The comparison is different if new_self is a torch tensor or an AST
            sgn_self = (new_self > 0).type(self.torch_dtype)
            pos_self = new_self * sgn_self
            neg_self = new_self * (sgn_self - 1)
            new_self = neg_self + pos_self

            # sgn_other is 1 when new_other is positive else it's 0
            # The comparison is different if new_other is a torch tensor or an AST
            sgn_other = (new_other > 0).type(self.torch_dtype)
            pos_other = new_other * sgn_other
            neg_other = new_other * (sgn_other - 1)
            new_other = neg_other + pos_other

            # If both have the same sign, sgn is 1 else it's 0
            # To be able to write sgn = 1 - (sgn_self - sgn_other) ** 2,
            # we would need to overload the __add__ for operators int and AST.
            sgn = -((sgn_self - sgn_other) ** 2) + 1
            changed_sign = True

            if cmd == "div":
                new_self *= self.base ** self.precision_fractional
        # Send it to the appropriate class and get the response
        response = getattr(new_self, cmd)(new_other)
        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response(
            cmd, response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )
        if not isinstance(other, (int, torch.Tensor, AdditiveSharingTensor)):
            if cmd == "mul":
                # If operation is mul, we need to truncate
                response = response.truncate(self.precision_fractional, check_sign=False)

            if changed_sign:
                # Give back its sign to response
                pos_res = response * sgn
                neg_res = response * (sgn - 1)
                response = neg_res + pos_res

        return response

    def mul(self, other):
        return self.mul_and_div(other, "mul")

    __mul__ = mul

    def __imul__(self, other):
        self.child = self.mul_and_div(other, "mul").child
        return self

    mul_ = __imul__

    def div(self, other):
        return self.mul_and_div(other, "div")

    __truediv__ = div

    def __itruediv__(self, other):
        self.child = self.mul_and_div(other, "div").child
        return self

    def pow(self, power):
        """
        Compute integer power of a number by recursion using mul

        This uses the following trick:
         - Divide power by 2 and multiply base to itself (if the power is even)
         - Decrement power by 1 to make it even and then follow the first step

        Args:
            power (int): the exponent supposed to be an integer > 0
        """
        if power < 0:
            raise RuntimeError("Negative integer powers are not allowed.")

        base = self

        result = None
        while power > 0:
            # If power is odd
            if power % 2 == 1:
                result = result * base if result is not None else base

            # Divide the power by 2
            power = power // 2
            # Multiply base to itself
            base = base * base

        return result

    __pow__ = pow

    def matmul(self, *args, **kwargs):
        """
        Hook manually matmul to add the truncation part which is inherent to multiplication
        in the fixed precision setting
        """

        other = args[0]

        if isinstance(other, FixedPrecisionTensor):
            assert (
                self.precision_fractional == other.precision_fractional
            ), "In matmul, all args should have the same precision_fractional"

        if isinstance(self.child, AdditiveSharingTensor) and isinstance(other.child, torch.Tensor):
            # If we try to matmul a FPT>AST with a FPT>torch.tensor,
            # we want to perform AST @ torch.tensor
            new_self = self.child
            new_args = (other,)
            new_kwargs = kwargs

        elif isinstance(other.child, AdditiveSharingTensor) and isinstance(
            self.child, torch.Tensor
        ):
            # If we try to matmul a FPT>torch.tensor with a FPT>AST,
            # we swap operators so that we do the same operation as above
            new_self = other.child
            new_args = (self,)
            new_kwargs = kwargs
        else:
            # Replace all syft tensor with their child attribute
            new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                "matmul", self, args, kwargs
            )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "matmul")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response(
            "matmul", response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        response = response.truncate(other.precision_fractional)

        return response

    __matmul__ = matmul
    mm = matmul

    def signum(self):
        """
        Calculation of signum function for a given tensor
        """
        sgn = (self > 0) - (self < 0)
        return sgn

    def modulus(self):
        """
        Calculation of modulus for a given tensor
        """
        return self.signum() * self

    def reciprocal(self, method="NR", nr_iters=10):
        r"""
        Calculate the reciprocal using the algorithm specified in the method args.
        Ref: https://github.com/facebookresearch/CrypTen

        Args:
            method:
            'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                    of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                    :math:`3*exp(-(x-.5)) + 0.003` as an initial guess by default
            'log' : Computes the reciprocal of the input from the observation that:
                    :math:`x^{-1} = exp(-log(x))`

            nr_iters:
                Number of iterations for `Newton-Raphson`
        Returns:
            Reciprocal of `self`
        """
        method = method.lower()

        if method == "nr":
            new_self = self.modulus()
            result = 3 * (0.5 - new_self).exp() + 0.003
            for i in range(nr_iters):
                result = 2 * result - result * result * new_self
            return result * self.signum()
        elif method == "newton":
            # it is assumed here that input values are taken in [-20, 20]
            x = None
            C = 20
            for i in range(80):
                if x is not None:
                    y = C + 1 - self * (x * x)
                    x = y * x / C
                else:
                    y = C + 1 - self
                    x = y / C
            return x
        elif method == "division":
            ones = self * 0 + 1
            return ones / self
        elif method == "log":
            new_self = self.modulus()
            return (-new_self.log()).exp() * self.signum()
        else:
            raise ValueError(f"Invalid method {method} given for reciprocal function")

    # Approximations:
    def inverse(self, iterations=8):
        """
        Computes an approximation of the matrix inversion using Newton-Schulz
        iterations
        """
        # TODO: should we add non-approximate version if self.child is a pure tensor?

        assert len(self.shape) >= 2, "Can't compute inverse on non-matrix"
        assert self.shape[-1] == self.shape[-2], "Must be batches of square matrices"

        inverse = (0.1 * torch.eye(self.shape[-1])).fix_prec(**self.get_class_attributes()).child

        for _ in range(iterations):
            inverse = 2 * inverse - inverse @ self @ inverse

        return inverse

    def exp(self, iterations=8):
        r"""
        Approximates the exponential function using a limit approximation:
        exp(x) = \lim_{n -> infty} (1 + x / n) ^ n

        Here we compute exp by choosing n = 2 ** d for some large d equal to
        iterations. We then compute (1 + x / n) once and square `d` times.

        Args:
            iterations (int): number of iterations for limit approximation

        Ref: https://github.com/LaRiffle/approximate-models
        """
        return (1 + self / 2 ** iterations) ** (2 ** iterations)

    def sign(self):
        return (self > 0) + (self < 0) * (-1)

    @staticmethod
    def _sigmoid_exp(tensor):
        """
        Implementation taken from FacebookResearch - CrypTen project

        Compute the sigmoid using the exp approximation
        sigmoid(x) = 1 / (1 + exp(-x))

        For stability:
            sigmoid(x) = (sigmoid(|x|) - 0.5) * sign(x) + 0.5

        Ref: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/#numerically_stable_sigmoid_function # noqa: E501

        Args:
            tensor (tensor): values where sigmoid should be approximated
        """

        sign = tensor.sign()

        # Make sure the elements are all positive
        x = tensor * sign
        ones = tensor * 0 + 1
        half = ones.div(2)
        result = (ones + (-ones * x).exp()).reciprocal(method="division")
        return (result - half) * sign + half

    @staticmethod
    def _sigmoid_maclaurin(tensor):
        """
        Approximates the sigmoid function using Maclaurin, with polynomial
        interpolation of degree 5 over [-8,8]
        NOTE: This method is faster but not as precise as "exp"
        Ref: https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/#approximating-sigmoid # noqa: E501

        Args:
            tensor (tensor): values where sigmoid should be approximated
        """

        weights = (
            torch.tensor([0.5, 1.91204779e-01, -4.58667307e-03, 4.20690803e-05])
            .fix_precision(**tensor.get_class_attributes())
            .child
        )
        degrees = [0, 1, 3, 5]

        # initiate with term of degree 0 to avoid errors with tensor ** 0
        one = tensor * 0 + 1
        result = one * weights[0]
        for i, d in enumerate(degrees[1:]):
            result += (tensor ** d) * weights[i + 1]

        return result

    @staticmethod
    def _sigmoid_chebyshev(tensor, maxval: int = 6, terms: int = 32):
        """
        Implementation taken from FacebookResearch - CrypTen project
        Computes the sigmoid function as
                 sigmoid(x) = (tanh(x /2) + 1) / 2

        Tanh is approximated using chebyshev polynomials
        Args:
             maxval (int): interval width used for tanh chebyshev polynomials
             terms (int): highest degree of Chebyshev polynomials for tanh.
                          Must be even and at least 6.
        """
        tanh_approx = tensor._tanh_chebyshev(tensor.div(2), maxval, terms)

        return tanh_approx.div(2) + 0.5

    def sigmoid(tensor, method="chebyshev"):
        """
        Approximates the sigmoid function using a given method

        Args:
            tensor: the fixed precision tensor
            method (str): (default = "chebyshev")
                Possible values: "exp", "maclaurin", "chebyshev"
        """

        sigmoid_f = getattr(tensor, f"_sigmoid_{method}")

        return sigmoid_f(tensor)

    def log(self, iterations=2, exp_iterations=8):
        """Approximates the natural logarithm using 8th order modified Householder iterations.
        Recall that Householder method is an algorithm to solve a non linear equation f(x) = 0.
        Here  f: x -> 1 - C * exp(-x)  with C = self

        Iterations are computed by:
            y_0 = some constant
            h = 1 - self * exp(-y_n)
            y_{n+1} = y_n - h * (1 + h / 2 + h^2 / 3 + h^3 / 6 + h^4 / 5 + h^5 / 7)

        Args:
            iterations (int): number of iterations for 6th order modified
                Householder approximation.
            exp_iterations (int): number of iterations for limit approximation of exp

        Ref: https://github.com/LaRiffle/approximate-models
        """

        y = self / 31 + 1.59 - 20 * (-2 * self - 1.4).exp(iterations=exp_iterations)

        # 6th order Householder iterations
        for i in range(iterations):
            h = [1 - self * (-y).refresh().exp(iterations=exp_iterations)]
            for i in range(1, 5):
                h.append(h[-1] * h[0])

            y -= h[0] * (1 + h[0] / 2 + h[1] / 3 + h[2] / 4 + h[3] / 5 + h[4] / 6)

        return y

    @staticmethod
    def _tanh_chebyshev(tensor, maxval: int = 6, terms: int = 32):
        r"""
        Implementation taken from FacebookResearch - CrypTen project
        Computes tanh via Chebyshev approximation with truncation.
          tanh(x) = \sum_{j=1}^terms c_{2j - 1} P_{2j - 1} (x / maxval)
          where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.
        The approximation is truncated to +/-1 outside [-maxval, maxval].

        Args:
            tensor (tensor): values where the tanh needs to be approximated
            maxval (int): interval width used for computing chebyshev polynomials
            terms (int): highest degree of Chebyshev polynomials.
                         Must be even and at least 6.

        More details can be found in the paper:
           Guo, Chuan and Hannun, Awni and Knott, Brian and van der Maaten,
           Laurens and Tygert, Mark and Zhu, Ruiyu,
           "Secure multiparty computations in floating-point arithmetic", Jan-2020
           Link: http://tygert.com/realcrypt.pdf

        """

        coeffs = syft.common.util.chebyshev_series(torch.tanh, maxval, terms)[1::2]
        coeffs = coeffs.fix_precision(**tensor.get_class_attributes())
        coeffs = coeffs.unsqueeze(1)

        value = torch.tensor(maxval).fix_precision(**tensor.get_class_attributes())
        tanh_polys = syft.common.util.chebyshev_polynomials(tensor.div(value.child), terms)
        tanh_polys_flipped = tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)

        out = tanh_polys_flipped.matmul(coeffs.child)

        # truncate outside [-maxval, maxval]
        gate_up = tensor > value
        gate_down = -tensor > value
        res = gate_up - gate_down
        out = out.squeeze(1) * (1 - gate_up - gate_down)
        out = res + out

        return out

    @staticmethod
    def _tanh_sigmoid(tensor):
        """
        Compute the tanh using the sigmoid approximation

        Args:
            tensor (tensor): values where tanh should be approximated
        """

        return 2 * torch.sigmoid(2 * tensor) - 1

    def tanh(tensor, method="chebyshev"):
        tanh_f = getattr(tensor, f"_tanh_{method}")

        return tanh_f(tensor)

    # Binary ops
    @overloaded.method
    def __gt__(self, _self, other):
        result = _self.__gt__(other)
        return result.type(self.torch_dtype) * self.base ** self.precision_fractional

    @overloaded.method
    def __ge__(self, _self, other):
        result = _self.__ge__(other)
        return result.type(self.torch_dtype) * self.base ** self.precision_fractional

    @overloaded.method
    def __lt__(self, _self, other):
        result = _self.__lt__(other)
        return result.type(self.torch_dtype) * self.base ** self.precision_fractional

    @overloaded.method
    def __le__(self, _self, other):
        result = _self.__le__(other)
        return result.type(self.torch_dtype) * self.base ** self.precision_fractional

    @overloaded.method
    def eq(self, _self, other):
        result = _self.eq(other)
        return result.type(self.torch_dtype) * self.base ** self.precision_fractional

    __eq__ = eq

    @overloaded.method
    def argmax(self, _self, **kwargs):
        result = _self.argmax(**kwargs)
        return result.long() * self.base ** self.precision_fractional

    @overloaded.method
    def argmin(self, _self, **kwargs):
        result = _self.argmin(**kwargs)
        return result.long() * self.base ** self.precision_fractional

    def var(self, unbiased=False, **kwargs):
        mu = self.mean(**kwargs)
        unbiased_self = self - mu
        mean = (unbiased_self * unbiased_self).mean(**kwargs)
        if unbiased:
            if kwargs.get("dim"):
                dim = kwargs["dim"]
                numel = self.shape[dim]
            else:
                numel = self.numel()
            return mean * numel / (numel - 1)
        else:
            return mean

    @staticmethod
    @overloaded.module
    def torch(module):
        def fmod(self, other):
            return self.__mod__(other)

        module.fmod = fmod

        def add(self, other):
            return self.__add__(other)

        module.add = add

        def sub(self, other):
            return self.__sub__(other)

        module.sub = sub

        def mul(self, other):
            return self.__mul__(other)

        module.mul = mul

        def div(self, other):
            return self.__truediv__(other)

        module.div = div

        def matmul(self, other):
            return self.matmul(other)

        module.matmul = matmul
        module.mm = matmul

        def addmm(bias, input_tensor, weight):
            matmul = input_tensor.matmul(weight)
            result = bias.add(matmul)
            return result

        module.addmm = addmm

        def inverse(self):
            return self.inverse()

        module.inverse = inverse

        def exp(tensor):
            return tensor.exp()

        module.exp = exp

        def sigmoid(tensor):
            return tensor.sigmoid()

        module.sigmoid = sigmoid

        def log(tensor):
            return tensor.log()

        module.log = log

        def tanh(tensor):
            return tensor.tanh()

        module.tanh = tanh

        def dot(self, other):
            return self.__mul__(other).sum()

        module.dot = dot

        # You can also overload functions in submodules!
        # Modules should be registered just like functions
        module.nn = nn  # Handles all the overloading properly

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
        <no self>, arguments[, kwargs_])
        :return: the response of the function command
        """
        cmd_name, _, args_, kwargs_ = command

        tensor = args_[0] if not isinstance(args_[0], (tuple, list)) else args_[0][0]

        # Check that the function has not been overwritten
        cmd = None
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd_name)
        except AttributeError:
            pass

        if cmd is not None:
            return cmd(*args_, **kwargs_)

        # Replace all FixedPrecisionTensor with their child attribute
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(
            cmd_name, args_, kwargs_
        )

        # build the new command
        new_command = (cmd_name, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back FixedPrecisionTensor on the tensors found in the response
        response = hook_args.hook_response(
            cmd_name, response, wrap_type=cls, wrap_args=tensor.get_class_attributes()
        )

        return response

    def share(self, *owners, protocol=None, field=None, dtype=None, crypto_provider=None):
        """
        Forward the .share() command to the child tensor, and reconstruct a new
        FixedPrecisionTensor since the command is not inplace and should return
        a new chain

        Args:
            *owners: the owners of the shares of the resulting AdditiveSharingTensor
            protocol: the crypto protocol used to perform the computations ('snn' or 'fss')
            field: the field size in which the share values live
            dtype: the dtype in which the share values live
            crypto_provider: the worker used to provide the crypto primitives used
                to perform some computations on AdditiveSharingTensors

        Returns:
            A FixedPrecisionTensor whose child has been shared
        """
        if dtype is None:
            dtype = self.dtype
        else:
            assert (
                dtype == self.dtype
            ), "When sharing a FixedPrecisionTensor, the dtype of the resulting AdditiveSharingTensor \
                must be the same as the one of the original tensor"

        tensor = FixedPrecisionTensor(owner=self.owner, **self.get_class_attributes())

        tensor.child = self.child.share(
            *owners, protocol=protocol, dtype=dtype, crypto_provider=crypto_provider, no_wrap=True
        )
        return tensor

    def share_(self, *args, **kwargs):
        """
        Performs an inplace call to share. The FixedPrecisionTensor returned is therefore the same,
        contrary to the classic share version
        """
        dtype = kwargs.get("dtype")
        if dtype is None:
            kwargs["dtype"] = self.dtype
        else:
            assert (
                dtype == self.dtype
            ), "When sharing a FixedPrecisionTensor, the dtype of the resulting AdditiveSharingTensor \
                must be the same as the one of the original tensor"
        kwargs.pop("no_wrap", None)
        self.child = self.child.share_(*args, no_wrap=True, **kwargs)
        return self

    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "FixedPrecisionTensor") -> tuple:
        """Takes the attributes of a FixedPrecisionTensor and saves them in a tuple.

        Args:
            worker: the worker doing the serialization
            tensor: a FixedPrecisionTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the fixed precision tensor.
        """
        chain = None
        if hasattr(tensor, "child"):
            chain = syft.serde.msgpack.serde._simplify(worker, tensor.child)

        return (
            syft.serde.msgpack.serde._simplify(worker, tensor.id),
            syft.serde.msgpack.serde._simplify(worker, tensor.field),
            tensor.dtype,
            tensor.base,
            tensor.precision_fractional,
            tensor.kappa,
            syft.serde.msgpack.serde._simplify(worker, tensor.tags),
            syft.serde.msgpack.serde._simplify(worker, tensor.description),
            chain,
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "FixedPrecisionTensor":
        """This function reconstructs a FixedPrecisionTensor given it's attributes in form
        of a tuple.

        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the FixedPrecisionTensor
        Returns:
            FixedPrecisionTensor: a FixedPrecisionTensor
        Examples:
            shared_tensor = detail(data)
        """

        (
            tensor_id,
            field,
            dtype,
            base,
            precision_fractional,
            kappa,
            tags,
            description,
            chain,
        ) = tensor_tuple

        tensor = FixedPrecisionTensor(
            owner=worker,
            id=syft.serde.msgpack.serde._detail(worker, tensor_id),
            field=syft.serde.msgpack.serde._detail(worker, field),
            dtype=dtype,
            base=base,
            precision_fractional=precision_fractional,
            kappa=kappa,
            tags=syft.serde.msgpack.serde._detail(worker, tags),
            description=syft.serde.msgpack.serde._detail(worker, description),
        )

        if chain is not None:
            chain = syft.serde.msgpack.serde._detail(worker, chain)
            tensor.child = chain

        return tensor

    @staticmethod
    def bufferize(worker, prec_tensor):
        """
        This method serializes FixedPrecisionTensor into FixedPrecisionTensorPB.

         Args:
            prec_tensor (FixedPrecisionTensor): input FixedPrecisionTensor to be serialized.

         Returns:
            proto_prec_tensor (FixedPrecisionTensorPB): serialized FixedPrecisionTensor
        """
        proto_prec_tensor = FixedPrecisionTensorPB()
        syft.serde.protobuf.proto.set_protobuf_id(proto_prec_tensor.id, prec_tensor.id)
        proto_prec_tensor.field = str(prec_tensor.field)
        proto_prec_tensor.dtype = prec_tensor.dtype
        proto_prec_tensor.base = prec_tensor.base
        proto_prec_tensor.kappa = prec_tensor.kappa
        proto_prec_tensor.precision_fractional = prec_tensor.precision_fractional
        for tag in prec_tensor.tags:
            proto_prec_tensor.tags.append(tag)
        proto_prec_tensor.description = prec_tensor.description
        if hasattr(prec_tensor, "child"):
            proto_prec_tensor.child.CopyFrom(
                syft.serde.protobuf.serde._bufferize(worker, prec_tensor.child)
            )

        return proto_prec_tensor

    @staticmethod
    def unbufferize(worker, proto_prec_tensor):
        """
        This method deserializes FixedPrecisionTensorPB into FixedPrecisionTensor.

        Args:
            proto_prec_tensor (FixedPrecisionTensorPB): input FixedPrecisionTensor to be
            deserialized.

        Returns:
            tensor (FixedPrecisionTensor): deserialized FixedPrecisionTensorPB
        """
        proto_id = syft.serde.protobuf.proto.get_protobuf_id(proto_prec_tensor.id)

        child = None
        if proto_prec_tensor.HasField("child"):
            child = syft.serde.protobuf.serde._unbufferize(worker, proto_prec_tensor.child)

        tensor = FixedPrecisionTensor(
            owner=worker,
            id=proto_id,
            field=proto_prec_tensor.field,
            dtype=proto_prec_tensor.dtype,
            base=proto_prec_tensor.base,
            precision_fractional=proto_prec_tensor.precision_fractional,
            kappa=proto_prec_tensor.kappa,
            tags=set(proto_prec_tensor.tags),
            description=proto_prec_tensor.description,
        )

        tensor.child = child
        return tensor

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for FixedPrecisionTensor.

        Returns:
            Protobuf schema for FixedPrecisionTensor.
        """
        return FixedPrecisionTensorPB


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(FixedPrecisionTensor)
