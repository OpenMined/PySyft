
import syft
import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
#from syft.frameworks.torch.tensors.interpreters.utils import hook

from . import functions
from . import gradients

def backwards_grad(grad_fn, in_grad=None):
    back_grad = grad_fn(in_grad)
    for next_grad_fn, next_grad in zip(grad_fn.next_functions, back_grad):
        backwards_grad(next_grad_fn, next_grad)

class AutogradTensor(AbstractTensor):
    """ A tensor that tracks operations to build a dynamic graph and backprops 
        through the graph to calculate gradients.
    """
    def __init__(self, data=None, requires_grad=True, owner=None, id=None):
        super().__init__()
        
        self.owner = owner
        self.id = id

        self.grad = None
        self.grad_fn = None
        self.child = data
        self.requires_grad = requires_grad
    
    def backward(self, grad=None):
        if grad is None:
            grad = torch.ones_like(self.child)
        backwards_grad(self.grad_fn, grad)

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value
    

    def add(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args = syft.frameworks.torch.hook_args.hook_method_args("add", self, args)
        
        result = getattr(new_self, "add")(*new_args, **kwargs)

        # Put back SyftTensor on the tensors found in the response
        result = syft.frameworks.torch.hook_args.hook_response(
            "add", result, wrap_type=type(self)
        )

        result.grad_fn = gradients.AddBackward(self, *args)

        return result

    def __add__(self, other):
        return self.add(other)

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a LoggingTensor,
        Perform some specific action (like logging) which depends of the
        instruction content, replace in the args all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a LoggingTensor on top of all tensors found in
        the response.
        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        # TODO: add kwargs in command
        cmd, _, args = command

        # Do what you have to
        #print("Logtensor logging function", cmd)

        # TODO: I can't manage the import issue, can you?
        # Replace all LoggingTensor with their child attribute
        new_args, new_type = syft.frameworks.torch.hook_args.hook_function_args(cmd, args)

        # build the new command
        new_command = (cmd, None, new_args)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back LoggingTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(cmd, response, wrap_type=cls)

        return response

def construct_autograd_method(name, gradient):
    
    def autograd_method(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args = syft.frameworks.torch.hook_args.hook_method_args(name, self, args)
        
        result = getattr(new_self, name)(*new_args, **kwargs)
        
        # Put back SyftTensor on the tensors found in the response
        result = syft.frameworks.torch.hook_args.hook_response(
            name, result, wrap_type=type(self)
        )

        result.grad_fn = getattr(gradients, gradient)(*args, **kwargs)

        return result

    return autograd_method
    
