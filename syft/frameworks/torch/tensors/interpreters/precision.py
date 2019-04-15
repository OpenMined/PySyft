import syft
import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded


class FixedPrecisionTensor(AbstractTensor):
    def __init__(
        self,
        owner=None,
        id=None,
        field: int = (2 ** 31) - 1,
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
        super().__init__(tags, description)

        self.owner = owner
        self.id = id
        self.child = None

        self.field = field
        self.base = base
        self.precision_fractional = precision_fractional
        self.kappa = kappa
        self.torch_max_value = torch.tensor([round(self.field / 2)])

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
        }

    def fix_precision(self):
        """This method encodes the .child object using fixed precision"""

        rational = self.child

        upscaled = (rational * self.base ** self.precision_fractional).long()
        field_element = upscaled % self.field
        field_element.owner = rational.owner

        self.child = field_element
        return self

    def float_precision(self):
        """this method returns a new tensor which has the same values as this
        one, encoded with floating point precision"""

        value = self.child.long() % self.field

        if len(value.size()) == 0:
            # raise TypeError("Can't decode empty tensor")
            return None

        gate = value.native_gt(self.torch_max_value).long()
        neg_nums = (value - self.field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums).float() / (self.base ** self.precision_fractional)

        return result

    def truncate(self, precision_fractional):
        truncation = self.base ** precision_fractional
        self.child /= truncation
        return self

    @overloaded.method
    def add(self, _self, *args, **kwargs):
        """Add two fixed precision tensors together.
        """
        response = getattr(_self, "add")(*args, **kwargs)

        return response % self.field

    __add__ = add

    @overloaded.method
    def t(self, _self, *args, **kwargs):
        """Transpose a tensor. Hooked is handled by the decorator"""
        response = getattr(_self, "t")(*args, **kwargs)

        return response

    def mul(self, *args, **kwargs):
        """
        Hook manually mul to add the truncation part which is inherent to multiplication
        in the fixed precision setting
        """
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
            "mul", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "mul")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            "mul", response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        other = args[0]

        assert (
            self.precision_fractional == other.precision_fractional
        ), "In mul, all args should have the same precision_fractional"

        return response.truncate(other.precision_fractional)

    __mul__ = mul

    def matmul(self, *args, **kwargs):
        """
        Hook manually matmul to add the truncation part which is inherent to multiplication
        in the fixed precision setting
        """
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
            "matmul", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "matmul")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            "matmul", response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        other = args[0]

        assert (
            self.precision_fractional == other.precision_fractional
        ), "In matmul, all args should have the same precision_fractional"

        return response.truncate(other.precision_fractional)

    __matmul__ = matmul

    @overloaded.method
    def __gt__(self, _self, other):
        result = _self.__gt__(other)
        return result * self.base ** self.precision_fractional

    @overloaded.method
    def __ge__(self, _self, other):
        result = _self.__ge__(other)
        return result * self.base ** self.precision_fractional

    @overloaded.method
    def __lt__(self, _self, other):
        result = _self.__lt__(other)
        return result * self.base ** self.precision_fractional

    @overloaded.method
    def __le__(self, _self, other):
        result = _self.__le__(other)
        return result * self.base ** self.precision_fractional

    @overloaded.method
    def eq(self, _self, other):
        result = _self.eq(other)
        return result * self.base ** self.precision_fractional

    @staticmethod
    @overloaded.module
    def torch(module):
        def mul(self, other):
            return self.__mul__(other)

        module.mul = mul

        def addmm(bias, input_tensor, weight):
            matmul = input_tensor.matmul(weight)
            result = bias.add(matmul)
            return result

        module.addmm = addmm

        # You can also overload functions in submodules!
        @overloaded.module
        def nn(module):
            """
            The syntax is the same, so @overloaded.module handles recursion
            Note that we don't need to add the @staticmethod decorator
            """

            @overloaded.module
            def functional(module):
                def linear(*args):
                    """
                    Un-hook the function to have its detailed behaviour
                    """
                    return torch.nn.functional.native_linear(*args)

                module.linear = linear

            module.functional = functional

        # Modules should be registered just like functions
        module.nn = nn

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

        tensor = args[0]

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # TODO: I can't manage the import issue, can you?
        # Replace all FixedPrecisionTensor with their child attribute
        new_args, new_kwargs, new_type = syft.frameworks.torch.hook_args.hook_function_args(
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

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        FixedPrecisionTensor which has also been shared."""
        class_attributes = self.get_class_attributes()
        return FixedPrecisionTensor(
            **class_attributes,
            owner=self.owner,
            tags=self.tags,
            description=self.description,
            id=self.id,
        ).on(self.child.get())

    def share(self, *owners, field=None, crypto_provider=None):
        self.child = self.child.share(*owners, field=field, crypto_provider=crypto_provider)
        return self
