# third party
import numpy as np

# syft relative
from ..passthrough import implements
from ..passthrough import inputs2child
from .tensor import AutogradTensor


@implements(AutogradTensor, np.max)
def npmax(*args, **kwargs):
    args, kwargs = inputs2child(*args, **kwargs)
    return np.max(*args, **kwargs)


@implements(AutogradTensor, np.min)
def npmin(*args, **kwargs):
    args, kwargs = inputs2child(*args, **kwargs)
    return np.min(*args, **kwargs)


@implements(AutogradTensor, np.expand_dims)
def expand_dims(*args, **kwargs):

    args, kwargs = inputs2child(*args, **kwargs)
    return np.expand_dims(*args, **kwargs)
