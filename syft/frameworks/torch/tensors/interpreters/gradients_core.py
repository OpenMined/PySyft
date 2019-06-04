# Module for implementing gradients used in the autograd system

__all__ = ["GradFunc"]


def forward_grad(tensor):
    ## tensor here should be an AutogradTensor or a Tensor where we can set .grad

    try:
        grad_fn = tensor.grad_fn
    except AttributeError:
        return None

    # If a tensor doesn't have a grad_fn already attached to it, that means
    # it's a leaf of the graph and we want to accumulate the gradient
    if grad_fn is None and tensor.requires_grad:
        return Accumulate(tensor)
    else:
        return grad_fn


class GradFunc:
    def __init__(self, *args):
        # This part builds our graph. It takes grad functions (if they exist)
        # from the input arguments and builds a tuple pointing to them. This way
        # we can use .next_functions to traverse through the entire graph.
        self.next_functions = tuple(
            filter(lambda x: x is not None, [forward_grad(arg) for arg in args])
        )
        self.result = None

    def gradient(self, grad):
        raise NotImplementedError

    def __setattr__(self, name, value):
        # Doing this because we want to pass in AutogradTensors so we can update
        # tensor.grad in Accumulate, but we also need native looking tensors for
        # the gradient operations in GradFuncs.
        try:
            value = value.child
        except AttributeError:
            pass

        object.__setattr__(self, name, value)

    def __call__(self, grad):
        return self.gradient(grad)

    def __repr__(self):
        return self.__class__.__name__


# Accumulate gradients at graph leafs
class Accumulate:
    def __init__(self, tensor):
        # Note that tensor here should be an AutogradTensor so we can update
        # .grad appropriately
        self.next_functions = []
        self.tensor = tensor

    def __call__(self, grad):
        if self.tensor.grad is not None:
            self.tensor.grad += grad
        else:
            self.tensor.grad = grad + 0
        return ()

    def __repr__(self):
        return self.__class__.__name__
