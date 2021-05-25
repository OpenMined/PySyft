# third party
import numpy as np

# syft relative
from .passthrough import implements
from .passthrough import inputs2child
from .tensor import Tensor


@implements(Tensor, np.mean)
def mean(array, axis=None, **kwargs):

    if axis is None:
        den = float(np.prod(array.shape))
    else:
        den = array.shape[axis]

    return array.sum(axis=axis) / den


@implements(Tensor, np.max)
def npmax(*args, **kwargs):
    args, kwargs = inputs2child(*args, **kwargs)
    return np.max(*args, **kwargs)


@implements(Tensor, np.min)
def npmin(*args, **kwargs):
    args, kwargs = inputs2child(*args, **kwargs)
    return np.min(*args, **kwargs)


@implements(Tensor, np.square)
def square(x):
    return x * x


@implements(Tensor, np.expand_dims)
def expand_dims(*args, **kwargs):
    args, kwargs = inputs2child(*args, **kwargs)
    return np.expand_dims(*args, **kwargs)
