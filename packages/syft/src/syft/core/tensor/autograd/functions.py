import numpy as np
from .tensor import AutogradTensor
from ..passthrough import inputs2child
from ..passthrough import implements

@implements(AutogradTensor, np.max)
def npmax(*args, **kwargs):
    print("maxing")
    args, kwargs = inputs2child(*args, **kwargs)
    return np.max(*args, **kwargs)

@implements(AutogradTensor, np.min)
def npmin(*args, **kwargs):
    print("mining")
    print(args)
    args, kwargs = inputs2child(*args, **kwargs)
    return np.min(*args, **kwargs)

@implements(AutogradTensor, np.expand_dims)
def expand_dims(*args, **kwargs):

    args, kwargs = inputs2child(*args, **kwargs)
    return np.expand_dims(*args, **kwargs)
