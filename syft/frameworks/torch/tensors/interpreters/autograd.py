from functools import wraps
import torch

import syft
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded
from . import gradients


def backwards_grad(grad_fn, in_grad=None):
    if grad_fn is None:
        raise ValueError(
            "The gradient for one of the command you used was not found. Check gradients.py "
            "to see if it's missing."
        )
    back_grad = grad_fn(in_grad)
    for next_grad_fn, next_grad in zip(grad_fn.next_functions, back_grad):
        backwards_grad(next_grad_fn, next_grad)


class AutogradTensor(AbstractTensor):
    """ A tensor that tracks operations to build a dynamic graph and backprops
        through the graph to calculate gradients.
    """

    def __init__(
        self, data=None, requires_grad=True, owner=None, id=None, preinitialize_grad=False, **kwargs
    ):
        super().__init__()

        self.owner = owner
        self.id = id

        self.child = data
        self.requires_grad = requires_grad
        self.preinitialize_grad = preinitialize_grad

        if preinitialize_grad:
            self.grad = data * 0
        else:
            self.grad = None
        self.grad_fn = None

    def backward(self, grad=None):
        if grad is None:
            # Build a torch tensor of ones with the same shape
            # And chain structure than self
            grad = self * 0 + 1
        backwards_grad(self.grad_fn, grad)

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_data):
        self.child = new_data.child
        return self

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    def attr(self, attr_name):
        if attr_name == "grad":
            return self.grad

        attr_val = self.child.attr(attr_name)
        return attr_val

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        result = self.add(other)
        self.child = result.child
        self.grad_fn = result.grad_fn

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        result = self.sub(other)
        self.child = result.child
        self.grad_fn = result.grad_fn

    def __mul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __pow__(self, power, **kwargs):
        return self.pow(power, **kwargs)

    def __truediv__(self, other):
        return self.div(other)

    @overloaded.method
    def __gt__(self, _self, other):
        return _self.__gt__(other)

    @overloaded.method
    def __ge__(self, _self, other):
        return _self.__ge__(other)

    @overloaded.method
    def __lt__(self, _self, other):
        return _self.__lt__(other)

    @overloaded.method
    def __le__(self, _self, other):
        return _self.__le__(other)

    @overloaded.method
    def eq(self, _self, other):
        return _self.eq(other)

    @overloaded.method
    def relu(self, self_):
        return self_.relu()

    def __getattribute__(self, name):
        # Automatically attaching gradient functions if they are defined in the
        # gradients module.
        grad_fn = getattr(gradients, name.capitalize() + "Backward", None)

        # print(f"getattribute {name}")
        if grad_fn is not None:

            def method_with_grad(*args, **kwargs):
                new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.unwrap_args_from_method(
                    name, self, args, kwargs
                )

                result = getattr(new_self, name)(*new_args, **new_kwargs)

                # Put back SyftTensor on the tensors found in the response
                result = syft.frameworks.torch.hook_args.hook_response(
                    name, result, wrap_type=type(self)
                )
                result.grad_fn = grad_fn(self, *args, **kwargs)
                result.grad_fn.result = result

                return result

            return method_with_grad
        else:
            return object.__getattribute__(self, name)

    @staticmethod
    @overloaded.module
    def torch(module):
        def add(self, other):
            return self.add(other)

        module.add = add

        def sub(self, other):
            return self.sub(other)

        module.sub = sub

        def mul(self, other):
            return self.mul(other)

        module.mul = mul

        def matmul(self, other):
            return self.matmul(other)

        module.matmul = matmul

        def div(self, other):
            return self.div(other)

        module.div = div

        def addmm(bias, input_tensor, weight):
            matmul = input_tensor.matmul(weight)
            result = bias.add(matmul)
            return result

        module.addmm = addmm

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

                def relu(tensor):
                    return tensor.relu()

                module.relu = relu

            module.functional = functional

        # Modules should be registered just like functions
        module.nn = nn

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a AutogradTensor,
        Perform some specific action (like logging) which depends of the
        instruction content, replace in the args all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a AutogradTensor on top of all tensors found in
        the response.
        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """

        cmd, _, args, kwargs = command

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # TODO: I can't manage the import issue, can you?
        # Replace all AutogradTensor with their child attribute
        new_args, new_kwargs, new_type = syft.frameworks.torch.hook_args.unwrap_args_from_function(
            cmd, args, kwargs
        )

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back AutogradTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(cmd, response, wrap_type=cls)

        return response

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        AutogradTensor which has also been shared."""
        self.child = self.child.get()
        # Remove the autograd node if a simple tensor is received
        if isinstance(self.child, torch.Tensor) and not self.child.is_wrapper:
            return self.child
        return self

    def float_precision(self):
        """Just a pass through. This is most commonly used when calling .float_precision() on a
        AutogradTensor which has also been shared."""
        self.child = self.child.float_precision()
        # Remove the autograd node if a simple tensor is received
        if isinstance(self.child, torch.Tensor) and not self.child.is_wrapper:
            return self.child
        return self

    @staticmethod
    def simplify(tensor: "AutogradTensor") -> tuple:
        """Takes the attributes of an AutogradTensor and saves them in a tuple.
            Or simply said, it serializes an AutogradTensor
        Args:
            tensor: an AutogradTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the AutogradTensor.
        """
        chain = syft.serde._simplify(tensor.child) if hasattr(tensor, "child") else None

        return (
            tensor.owner,
            syft.serde._simplify(tensor.id),
            chain,
            tensor.requires_grad,
            tensor.preinitialize_grad,
            tensor.grad_fn,
            # tensor.local_autograd,
            syft.serde._simplify(tensor.tags),
            syft.serde._simplify(tensor.description),
        )

    @staticmethod
    def detail(worker: AbstractTensor, tensor_tuple: tuple) -> "AutogradTensor":
        """
            This function reconstructs (deserializes) an AutogradTensors given its attributes in form of a tuple.
            Args:
                worker: the worker doing the deserialization
                tensor_tuple: a tuple holding the attributes of the AutogradTensor
            Returns:
                AutogradTensor: an AutogradTensor
            Examples:
                shared_tensor = detail(data)
            """
        owner, tensor_id, chain, requires_grad, preinitialize_grad, grad_fn, tags, description = (
            tensor_tuple
        )

        if chain is not None:
            chain = syft.serde._detail(worker, chain)

        tensor = AutogradTensor(
            owner=owner,
            id=syft.serde._detail(worker, tensor_id),
            requires_grad=requires_grad,  # ADDED!
            preinitialize_grad=preinitialize_grad,
            grad_fn=grad_fn,
            # local_autograd=local_autograd,
            data=chain,  # pass the de-serialized data
            tags=syft.serde._detail(worker, tags),
            description=syft.serde._detail(worker, description),
        )

        return tensor
