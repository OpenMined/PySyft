from functools import wraps
import torch

import syft
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters import PointerTensor
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
        self,
        data=None,
        requires_grad=True,
        owner=None,
        id=None,
        parent=None,
        preinitialize_grad=False,
        **kwargs,
    ):
        super().__init__()

        self.owner = owner
        self.id = id
        self.parent = parent

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
            self.child.setattr("grad", value)
            self._grad = value.wrap()
        else:
            self._grad = value

        if self.parent is not None:
            # self.parent is a weakref
            self.parent().grad = self._grad

    def attr(self, attr_name):
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

    def __mul__(self, other):
        return self.mul(other)

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
