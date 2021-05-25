# third party
import numpy as np
import torch as th

# syft relative
from .ancestors import AutogradTensorAncestor
from .ancestors import SingleEntityPhiTensorAncestor
from .passthrough import HANDLED_FUNCTIONS
from .passthrough import PassthroughTensor


class Tensor(PassthroughTensor, AutogradTensorAncestor, SingleEntityPhiTensorAncestor):
    def __init__(self, child):
        """data must be a list of numpy array"""

        if isinstance(child, list):
            child = np.array(child)

        if isinstance(child, th.Tensor):
            child = child.numpy()

        if not isinstance(child, PassthroughTensor) and not isinstance(
            child, np.ndarray
        ):
            raise Exception("Data must be list or nd.array")

        super().__init__(child=child)

    def __array_function__(self, func, types, args, kwargs):
        #         args, kwargs = inputs2child(*args, **kwargs)

        # Note: this allows subclasses that don't override
        # __array_function__ to handle PassthroughTensor objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        if func in HANDLED_FUNCTIONS[self.__class__]:
            return HANDLED_FUNCTIONS[self.__class__][func](*args, **kwargs)
        else:
            return self.__class__(func(*args, **kwargs))
