import numpy as np

from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.frameworks.torch.tensors.interpreters.hook import HookedTensor


class NumpyTensor(HookedTensor):
    """NumpyTensor is a tensor which seeks to wrap the Numpy API with the PyTorch tensor API.
    This is useful because Numpy can offer a wide range of existing functionality ranging from
    large precision, custom scalar types, and polynomial arithmetic.
    """

    def __init__(
        self, numpy_tensor=None, owner=None, id=None, tags=None, description=None, verbose=False
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

        if isinstance(numpy_tensor, list):
            numpy_tensor = np.array(numpy_tensor)

        self.child = numpy_tensor

    @overloaded.method
    def mm(self, _self, other):
        return _self.dot(other)

    @overloaded.method
    def transpose(self, _self, *dims):
        # TODO: the semantics of the .transpose() dimensions are a bit different
        # for Numpy than they are for PyTorch. Fix this.
        # Related: https://github.com/pytorch/pytorch/issues/7609
        return _self.transpose(*reversed(dims))


def create_numpy_tensor(numpy_tensor):
    return NumpyTensor(numpy_tensor).wrap()


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(NumpyTensor)
