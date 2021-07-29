# third party
import numpy as np

# relative
# syft relative
from .passthrough import implements
from .passthrough import inputs2child
from .tensor import Tensor


@implements(Tensor, np.mean)
def mean(array, axis=None, **kwargs) -> Tensor:
    if axis is None:
        den = float(np.prod(array.shape))
    else:
        den = array.shape[axis]

    return array.sum(axis=axis) / den


@implements(Tensor, np.max)
def npmax(*args, **kwargs) -> Tensor:
    args, kwargs = inputs2child(*args, **kwargs)
    return np.max(*args, **kwargs)


@implements(Tensor, np.min)
def npmin(*args, **kwargs) -> Tensor:
    args, kwargs = inputs2child(*args, **kwargs)
    return np.min(*args, **kwargs)


@implements(Tensor, np.square)
def square(x) -> Tensor:
    return x * x


@implements(Tensor, np.expand_dims)
def expand_dims(*args, **kwargs) -> Tensor:
    args, kwargs = inputs2child(*args, **kwargs)
    return Tensor(np.expand_dims(*args, **kwargs))


@implements(Tensor, np.multiply)
def multiply(a, b) -> Tensor:
    if isinstance(a, Tensor):
        result = a.__mul__(b)
        if result is not NotImplementedError:
            return result

    if isinstance(b, Tensor):
        result = b.__rmul__(a)
        if result is not NotImplementedError:
            return result

    return TypeError(f"Can't multiply {type(a)} with {type(b)}")
