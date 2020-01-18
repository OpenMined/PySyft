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

    def attr(self, name):
        if(name == "data"):
            return self.wrap()

    @overloaded.method
    def expand(self, _self, *dims):

        if (len(dims) != len(_self.shape)):
            raise Exception(".expand() must be called with the same number of dims as the tensor on which it is called")

        n_diff = 0
        diff_i = 0
        for i in range(len(_self.shape)):
            if (_self.shape[i] != dims[i]):
                diff_i = i
                n_diff += 1

        if (n_diff > 1):
            raise Exception("You can only call .expand() with one dim different")

        return np.repeat(_self, dims[diff_i], axis=diff_i).reshape(dims)

    @overloaded.method
    def view(self, _self, *dims):
        return _self.reshape(*dims)

    @overloaded.method
    def unsqueeze(self, _self, axis):
        return np.expand_dims(_self, axis=axis)

    @overloaded.method
    def t(self, _self, *args, **kwargs):
        return _self.transpose(*args, **kwargs)

    @overloaded.method
    def mm(self, _self, other):
        return _self.dot(other)

def create_numpy_tensor(numpy_tensor):
    result = NumpyTensor(numpy_tensor).wrap()
    result.is_wrapper = True
    return result

### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(NumpyTensor)
