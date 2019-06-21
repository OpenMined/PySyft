from functools import wraps
import torch

import syft
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded
from . import gradients


def backwards_grad(grad_fn, in_grad=None):
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
            grad = torch.ones(self.child.shape)
        backwards_grad(self.grad_fn, grad)

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if value is not None:
            assert isinstance(value, AutogradTensor)
            self._grad = value.child
        else:
            self._grad = value

    def attr(self, attr_name):
        if attr_name == "grad":
            return self.grad

        attr_val = self.child.attr(attr_name)
        return attr_val

    def __getattribute__(self, name):
        # Automatically attaching gradient functions if they are defined in the
        # gradients module.
        grad_fn = getattr(gradients, name.capitalize() + "Backward", None)

        # print(f"getattribute {name}")
        if grad_fn is not None:

            def method_with_grad(*args, **kwargs):
                new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
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

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __pow__(self, power, **kwargs):
        return self.pow(power, **kwargs)

    @staticmethod
    @overloaded.module
    def torch(module):
        def add(self, other):
            return self.__add__(other)

        module.add = add

        def sub(self, other):
            return self.__sub__(other)

        module.sub = sub

        def mul(self, other):
            return self.__mul__(other)

        module.mul = mul

        def matmul(self, other):
            return self.matmul(other)

        module.matmul = matmul

        def addmm(bias, input_tensor, weight):
            matmul = input_tensor.matmul(weight)
            result = bias.add(matmul)
            return result

        module.addmm = addmm

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
        new_args, new_kwargs, new_type = syft.frameworks.torch.hook_args.hook_function_args(
            cmd, args, kwargs
        )

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back AutogradTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(cmd, response, wrap_type=cls)

        return response
