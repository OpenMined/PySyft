# Module for implementing gradients used in the autograd system

import syft
from syft.workers.abstract import AbstractWorker
from syft.generic.abstract.syft_serializable import SyftSerializable

from . import gradients

__all__ = ["GradFunc", "apply_dim_transformations"]


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


class GradFunc(SyftSerializable):
    def __init__(self, *args):
        # This part builds our graph. It takes grad functions (if they exist)
        # from the input arguments and builds a tuple pointing to them. This way
        # we can use .next_functions to traverse through the entire graph.
        # _attributes is a private list that records all the attributes (required for
        # serialization/deserialization)

        self.next_functions = tuple(
            filter(lambda x: x is not None, (forward_grad(arg) for arg in args))
        )
        # self.result = None TODO: Broken as of Garbage Collection for `AutoGradTensor` (#3387)
        self._attributes = []

    def gradient(self, grad):
        raise NotImplementedError

    def __call__(self, grad):
        return self.gradient(grad)

    def __repr__(self):
        return self.__class__.__name__

    def __setattr__(self, name, value):
        self.__dict__[name] = value

        # add attributes of grad function to _attributes list
        # essential for serialization and deserialization
        if name not in {"next_functions", "result", "_attributes"}:
            self._attributes.append(value)

    @staticmethod
    def simplify(worker: AbstractWorker, grad_fn) -> tuple:
        """Takes the attributes of a grad_fn object and saves them in a tuple
            Every gradient function class that extends `GradFunc` uses this function to
            simplify the attributes.

        Args:
            grad_fn: gradient function object (AddBackward, SubBackward, etc.)

        Returns:
            grad_fn_attrs: a tuple containing all the simplified attributes
            of gradient function
        """

        # Add class name of the grad_fn object at the begining of attributes list.
        # Essential while deserializing the object
        cls = grad_fn.__class__.__name__
        grad_fn_attrs = [cls] + grad_fn._attributes

        # Simplify the attributes list
        grad_fn_attrs = syft.serde.msgpack.serde._simplify(worker, grad_fn_attrs)

        return grad_fn_attrs

    @staticmethod
    def detail(worker: AbstractWorker, gradfn_tuple):
        """This function reconstructs (deserializes) the gradient function object,
         given its attributes in the form of a tuple

         Args:
            gradfn_tuple: a tuple containing all the simplified attributes of a
            grad function (along with the class name at begining)

        Returns:
            A correct gradient function object

        """
        # Detail and extract the class name and attributes from the tuple
        grad_fn_attrs = []
        cls, *grad_fn_attrs = syft.serde.msgpack.serde._detail(worker, gradfn_tuple)

        if cls == "GradFunc":
            cls = GradFunc
        else:
            cls = getattr(gradients, cls)

        return cls(*grad_fn_attrs)


# Accumulate gradients at graph leafs
class Accumulate:
    def __init__(self, tensor):
        # Note that tensor here should be an AutogradTensor so we can update
        # .grad appropriately
        self.next_functions = []
        self.tensor = tensor

    def __call__(self, grad):
        if self.tensor.grad is not None:
            self.tensor.grad.add_(grad.child)
        else:
            self.tensor.grad = grad.child.copy()
        return ()

    def __repr__(self):
        return self.__class__.__name__


def get_mismatch_dims(
    x_shape, y_shape, x_squash_dims, x_squash_keep_dims, y_squash_keep_dims, current_dim=None
):
    """
    Given two tensors shape, build three lists referencing specific dimension indices where
    the dimension for the two tensors is not the same:

    Note that we assume len(x_shape) <= len(y_shape) and inverse x and y if necessary.

    1. If one tensor has a longer shape (so lives in higher dimension)(so it is y because
    len(x_shape) <= len(y_shape)), then all the extra dimensions indices end up in x_squash_dims.
    Indeed, x will be automatically expanded in all the extra dimensions at forward operation time
    to match the tensor with the highest dimension. Example: tensor([1]) + tensor([[2], [3]]) is
    rewritten by torch as tensor([[1], [1]]) + tensor([[2], [3]]). So we need to remove all these
    dimensions of expansion to get back the gradient of x.

    2. If x or y has a 1 in some dimension and doesn't match the other tensor
    (ex: torch.Size([3, 7, 5]) and torch.Size([3, 1, 5])), then when an operation is called the
    tensor is expanded on this dimension, so we need to register this information to do the same
    squashing as before but we keep the dimension this time. This apply to both tensors and hence
    we use the lists x_squash_keep_dims and y_squash_keep_dims

    3. current_dim is used to keep count of the dimension, as we iterate on the dimension starting
    from the highest (the most right one) down to zero. Generally, remember that the shape tuple
    should match when align on the right, unless there is a 1 instead of the other value: shapes
    (5,4,3,2) and (4,1,2) are compatible, (5,4,3,2) and (5,4,2,2) are not.
    """
    # Initialize the dimension if needed
    if current_dim is None:
        current_dim = max(len(x_shape), len(y_shape)) - 1
        # Inverse x and y if needed to satisfy len(x_shape) <= len(y_shape)
        if len(x_shape) > len(y_shape):
            return get_mismatch_dims(
                y_shape, x_shape, x_squash_dims, x_squash_keep_dims, y_squash_keep_dims, current_dim
            )
    if len(y_shape) == 0:  # implies also len(x_shape) == 0
        return
    elif len(x_shape) == 0:  # elif implies now len(y_shape) > 0 (See case 1.)
        x_squash_dims.append(current_dim)
        get_mismatch_dims(
            x_shape,
            y_shape[:-1],
            x_squash_dims,
            x_squash_keep_dims,
            y_squash_keep_dims,
            current_dim - 1,
        )
    else:  # implies len(x_shape) > 0 and len(y_shape) > 0 (See case 2.)
        if x_shape[-1] != y_shape[-1]:
            if x_shape[-1] == 1:
                x_squash_keep_dims.append(current_dim)
            elif y_shape[-1] == 1:
                y_squash_keep_dims.append(current_dim)
        get_mismatch_dims(
            x_shape[:-1],
            y_shape[:-1],
            x_squash_dims,
            x_squash_keep_dims,
            y_squash_keep_dims,
            current_dim - 1,
        )


def apply_dim_transformations(grad_self, grad_other, self_shape, other_shape):
    """
    Given computed gradients and initial shapes, reshape the gradients to match the
    initial shapes by reverse engineering the expansion operations made by PyTorch
    when operating two tensors with different shapes.

    Args:
        grad_self: computed gradient for self
        grad_other: computed gradient for other
        self_shape: initial shape for self
        other_shape: initial shape for other

    Returns:
        grad_self, grad_other with the proper shape
    """
    short_squash_dims = []
    short_squash_keep_dims = []
    long_squash_keep_dims = []
    get_mismatch_dims(
        self_shape, other_shape, short_squash_dims, short_squash_keep_dims, long_squash_keep_dims
    )

    # Flip self and other if needed
    if len(self_shape) <= len(other_shape):
        short_grad, long_grad = grad_self.child, grad_other.child
    else:
        short_grad, long_grad = grad_other.child, grad_self.child

    # Reduce dimensions by summations
    if short_squash_keep_dims:
        short_grad = short_grad.sum(dim=short_squash_keep_dims, keepdim=True)
    if short_squash_dims:
        short_grad = short_grad.sum(dim=short_squash_dims)
    if long_squash_keep_dims:
        long_grad = long_grad.sum(dim=long_squash_keep_dims, keepdim=True)

    # Reverse the flip
    if len(self_shape) <= len(other_shape):
        grad_self.child, grad_other.child = short_grad, long_grad
    else:
        grad_self.child, grad_other.child = long_grad, short_grad

    return grad_self, grad_other
