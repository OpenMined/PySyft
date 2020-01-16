from syft.generic.frameworks.hook.hook_args import (
    register_type_rule,
    register_forward_func,
    register_backward_func,
    one,
)

from syft.generic.tensor import AbstractTensor


class NumpyTensor(AbstractTensor):
    """NumpyTensor is a tensor which seeks to wrap the Numpy API with the PyTorch tensor API.
    This is useful because Numpy can offer a wide range of existing functionality ranging from
    large precision, custom scalar types, and polynomial arithmetic.
    """

    def __init__(
        self,
        numpy_tensor=None,
        owner=None,
        id=None,
        tags=None,
        description=None,
        verbose=False,
    ):
        """Initializes a NumpyTensor.

        Args:
            numpy_tensor (np.array): The numpy array which this tensor should wrap.
            owner (BaseWorker): An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id (str or int): An optional string or integer id of the LargePrecisionTensor.
            tags (list): list of tags for searching.
            description (str): a description of this tensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.verbose = verbose
        self.child = numpy_tensor

    def __add__(self, other):
        return NumpyTensor(numpy_tensor=self.child + other.child)

    def __sub__(self, other):
        return NumpyTensor(numpy_tensor=self.child - other.child)

    def __mul__(self, other):
        return NumpyTensor(numpy_tensor=self.child * other.child)

    def __truediv__(self, other):
        return NumpyTensor(numpy_tensor=self.child / other.child)

    def dot(self, other):
        return NumpyTensor(numpy_tensor=self.child.dot(other.child))

    def mm(self, other):
        return NumpyTensor(numpy_tensor=self.child.dot(other.child))

    def __matmul__(self, other):
        return NumpyTensor(numpy_tensor=self.child.dot(other.child))

    def transpose(self, *dims):
        # TODO: the semantics of the .transpose() dimensions are a bit different
        # for Numpy than they are for PyTorch. Fix this.
        # Related: https://github.com/pytorch/pytorch/issues/7609
        return NumpyTensor(numpy_tensor=self.child.transpose(*reversed(dims)))


### Register the tensor with hook_args.py ###
register_type_rule({NumpyTensor: one})
register_forward_func({NumpyTensor: lambda i: NumpyTensor._forward_func(i)})
register_backward_func(
    {NumpyTensor: lambda i, **kwargs: NumpyTensor._backward_func(i, **kwargs)}
)
