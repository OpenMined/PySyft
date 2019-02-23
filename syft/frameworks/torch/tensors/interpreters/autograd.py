import syft
import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.utils import hook

from . import gradients

def backwards_grad(grad_fn, in_grad=None):
    back_grad = grad_fn(in_grad)
    for next_grad_fn in grad_fn.next_functions:
        backwards_grad(next_grad_fn, back_grad)

class AutogradTensor(AbstractTensor):
    """A tensor that tracks operations to build a dynamic graph and backprops 
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
        print("My backward")
        if grad is None:
            grad = torch.ones_like(self.child)
        backwards_grad(self.grad_fn, grad)
    

    def __add__(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args = syft.frameworks.torch.hook_args.hook_method_args("__add__", self, args)
        other = new_args[0]

        # Send it to the appropriate class and get the response
        # response = getattr(new_self, "add")(*new_args, **kwargs)
        #result = AutogradTensor(new_self + other, requires_grad=new_self.requires_grad)
        result = new_self + other
        # Put back SyftTensor on the tensors found in the response
        result = syft.frameworks.torch.hook_args.hook_response(
            "__add__", result, wrap_type=type(self)
        )

        grad_fn = gradients.GradAdd()
        grad_fn.next_functions = (gradients.Accumulate(self) if self.grad_fn is None else self.grad_fn, 
                                  gradients.Accumulate(args[0]) if args[0].grad_fn is None else args[0].grad_fn)
        result.grad_fn = grad_fn

        return result

    @property
    def grad(self):
        print("Getting grad")
        return self._grad

    @grad.setter
    def grad(self, value):
        print('Setting gradient')
        self._grad = value
    
