# Module for implementing gradients used in the autograd system
from torch import Tensor

__all__ = ["Tensor", "GradFunc", "Accumulate"]

def forward_grad(tensor):
    try:
        grad_fn = tensor.grad_fn
    except AttributeError:
        return None
    return Accumulate(tensor) if grad_fn is None else grad_fn


class GradFunc:
    def __init__(self, *args):
        # This part builds our graph. It takes grad functions (if they exist)
        # from the input arguments and builds a list pointing to them. This way
        # we can use .next_functions to traverse through the entire graph.
        self.next_functions = tuple(filter(lambda x: x is not None, 
                                    [forward_grad(data) for data in args]))

    def gradient(self, grad):
        raise NotImplementedError

    @property
    def size(self):
        return len(self.next_functions)

    def __call__(self, grad):
        back_grad = self.gradient(grad)
        return back_grad

    def __repr__(self):
        return self.__class__.__name__


class Accumulate(GradFunc):
    def __init__(self, tensor):
        self.next_functions = []
        self.tensor = tensor
        
    def gradient(self, grad):
        if self.tensor.grad is not None:
            self.tensor.grad += grad
        else:
            self.tensor.grad = grad + 0

        return ()