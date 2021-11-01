# third party
import numpy as np

# relative
from ..util import implements
from ..util import inputs2child
from .tensor import AutogradTensor


@implements(AutogradTensor, np.max)
def npmax(*args: AutogradTensor, **kwargs: AutogradTensor) -> AutogradTensor:
    args, kwargs = inputs2child(*args, **kwargs)
    return np.max(*args, **kwargs)


@implements(AutogradTensor, np.min)
def npmin(*args: AutogradTensor, **kwargs: AutogradTensor) -> AutogradTensor:
    args, kwargs = inputs2child(*args, **kwargs)
    return np.min(*args, **kwargs)


@implements(AutogradTensor, np.expand_dims)
def expand_dims(*args: AutogradTensor, **kwargs: AutogradTensor) -> AutogradTensor:
    requires_grad = args[0].requires_grad
    output_type = args[0].__class__
    args, kwargs = inputs2child(*args, **kwargs)
    return output_type(np.expand_dims(*args, **kwargs), requires_grad=requires_grad)


@implements(AutogradTensor, np.multiply)
def multiply(a: AutogradTensor, b: AutogradTensor) -> AutogradTensor:
    return AutogradTensor(a * b)
