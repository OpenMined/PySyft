import torch

import syft
from syft.generic.abstract.tensor import AbstractTensor
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.generic.frameworks.hook.hook_args import (
    get_child,
    register_backward_func,
    register_forward_func,
    register_type_rule,
    one,
)
from syft.workers.abstract import AbstractWorker
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
    """A tensor that tracks operations to build a dynamic graph and backprops
    through the graph to calculate gradients.
    """

    def __init__(
        self, data=None, requires_grad=True, owner=None, id=None, preinitialize_grad=False, **kwargs
    ):
        super().__init__(
            id=id, owner=owner, tags=kwargs.get("tags"), description=kwargs.get("description")
        )

        self.child = data
        self.requires_grad = requires_grad
        self.preinitialize_grad = preinitialize_grad

        if preinitialize_grad:
            self.grad = data * 0
        else:
            self.grad = None

        self.grad_fn = kwargs.get("grad_fn")

    def backward(self, grad=None):
        if grad is None:
            # Build a torch tensor of ones with the same shape
            # And chain structure than self
            grad = self * 0 + 1
        backwards_grad(self.grad_fn, grad)

    @property
    def data(self):
        # TODO why is that? Normally .data is detached from autograd
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
        if isinstance(self, AutogradTensor) and not isinstance(other, AutogradTensor):
            other = AutogradTensor(requires_grad=False).on(other, wrap=False)
        return self.add(other)

    def __iadd__(self, other):
        result = self.add(other)
        self.child = result.child
        self.grad_fn = result.grad_fn
        return self

    def __sub__(self, other):
        if isinstance(self, AutogradTensor) and not isinstance(other, AutogradTensor):
            other = AutogradTensor(requires_grad=False).on(other, wrap=False)
        return self.sub(other)

    def __isub__(self, other):
        result = self.sub(other)
        self.child = result.child
        self.grad_fn = result.grad_fn
        return self

    def __mul__(self, other):
        if isinstance(self, AutogradTensor) and not isinstance(other, AutogradTensor):
            other = AutogradTensor(requires_grad=False).on(other, wrap=False)
        return self.mul(other)

    def __neg__(self):
        return self.neg()

    def __matmul__(self, other):
        if isinstance(self, AutogradTensor) and not isinstance(other, AutogradTensor):
            other = AutogradTensor(requires_grad=False).on(other, wrap=False)

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
    def relu(self, self_, **kwargs):
        return self_.relu()

    def __getattribute__(self, name):
        # Automatically attaching gradient functions if they are defined in the
        # gradients module.
        grad_fn = getattr(gradients, name.capitalize() + "Backward", None)

        # print(f"getattribute {name}")
        if grad_fn is not None:

            def method_with_grad(*args, **kwargs):
                new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                    name, self, args, kwargs
                )

                result = getattr(new_self, name)(*new_args, **new_kwargs)

                # Put back SyftTensor on the tensors found in the response
                result = hook_args.hook_response(name, result, wrap_type=type(self))
                result.grad_fn = grad_fn(self, *args, **kwargs)

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

        def neg(self):
            return self.neg()

        module.neg = neg

        def log(self):
            """Overriding torch's log method."""
            return self.log()

        module.log = log

        def exp(self):
            """Overriding torch's exp function."""
            return self.exp()

        module.exp = exp

        def sum(self, **kwargs):
            """Overriding torch's sum function."""
            return self.sum(**kwargs)

        module.sum = sum

        def mean(self, **kwargs):
            return self.mean(**kwargs)

        module.mean = mean

        def matmul(self, other):
            return self.matmul(other)

        module.matmul = matmul

        def div(self, other):
            return self.div(other)

        module.div = div

        def addmm(bias, input_tensor, weight):
            if not isinstance(input_tensor, AutogradTensor):
                input_tensor = AutogradTensor(requires_grad=False).on(input_tensor, wrap=False)

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

                def relu(tensor, **kwargs):
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
        <no self>, arguments[, kwargs_])
        :return: the response of the function command
        """

        cmd_name, _, args_, kwargs_ = command

        # Check that the function has not been overwritten
        cmd = None
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd_name)
        except AttributeError:
            pass

        if cmd is not None:
            return cmd(*args_, **kwargs_)

        # Replace all AutogradTensor with their child attribute
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(
            cmd_name, args_, kwargs_
        )

        # build the new command
        new_command = (cmd_name, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back AutogradTensor on the tensors found in the response
        response = hook_args.hook_response(cmd_name, response, wrap_type=cls)

        return response

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        AutogradTensor which has also been shared."""
        tensor = self.child.get()

        if isinstance(tensor, torch.Tensor):
            # Remove the autograd node if a simple tensor is received
            if not tensor.is_wrapper:
                return tensor
            # If it's a wrapper, then insert the autograd under the wrapper
            else:
                self.child = tensor.child
                tensor.child = self
                return tensor

        self.child = tensor
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
    def simplify(worker: AbstractWorker, tensor: "AutogradTensor") -> tuple:
        """Takes the attributes of an AutogradTensor and saves them in a tuple.
            Or simply said, it serializes an AutogradTensor
        Args:
            tensor: an AutogradTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the AutogradTensor.
        """
        chain = (
            syft.serde.msgpack.serde._simplify(worker, tensor.child)
            if hasattr(tensor, "child")
            else None
        )

        return (
            syft.serde.msgpack.serde._simplify(worker, tensor.id),
            chain,
            tensor.requires_grad,
            tensor.preinitialize_grad,
            syft.serde.msgpack.serde._simplify(worker, tensor.grad_fn),
            # tensor.local_autograd,
            syft.serde.msgpack.serde._simplify(worker, tensor.tags),
            syft.serde.msgpack.serde._simplify(worker, tensor.description),
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "AutogradTensor":
        """
            This function reconstructs (deserializes) an AutogradTensor given its
        attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the AutogradTensor
        Returns:
            AutogradTensor: an AutogradTensor
        Examples:
            shared_tensor = detail(data)
        """
        (
            tensor_id,
            chain,
            requires_grad,
            preinitialize_grad,
            grad_fn,
            tags,
            description,
        ) = tensor_tuple

        if chain is not None:
            chain = syft.serde.msgpack.serde._detail(worker, chain)

        tensor = AutogradTensor(
            owner=worker,
            id=syft.serde.msgpack.serde._detail(worker, tensor_id),
            requires_grad=requires_grad,  # ADDED!
            preinitialize_grad=preinitialize_grad,
            # local_autograd=local_autograd,
            grad_fn=syft.serde.msgpack.serde._detail(worker, grad_fn),
            data=chain,  # pass the de-serialized data
            tags=syft.serde.msgpack.serde._detail(worker, tags),
            description=syft.serde.msgpack.serde._detail(worker, description),
        )

        return tensor


register_type_rule({AutogradTensor: one})
register_forward_func({AutogradTensor: get_child})
register_backward_func(
    {AutogradTensor: lambda i, **kwargs: AutogradTensor(data=i).on(i, wrap=False)}
)
