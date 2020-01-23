import numpy as np

from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.generic.tensor import AbstractTensor


class LazyTensor(AbstractTensor):

    def __init__(
            self, numpy_tensor=None, owner=None, id=None, tags=None, description=None, verbose=False
    ):
        """Initializes a LazyTensor.

        Args:
            numpy_tensor (np.array): The numpy array which this tensor should wrap.
            owner (BaseWorker): An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id (str or int): An optional string or integer id of the LargePrecisionTensor.
            tags (list): list of tags for searching.
            description (str): a description of this tensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.todos = list()

    def log_softmax(self, *args, **kwargs):
        self.todos.append(('log_softmax', args, kwargs))
        return self

    def execute(self):
        result = self
        for method_name, args, kwargs in self.todos:
            result = self.child.__getattribute__(method_name)(*args, **kwargs)
        return result




### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(LazyTensor)
