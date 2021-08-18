# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
import numpy as np

# relative
from .passthrough import implements
from .passthrough import inputs2child
from .tensor import Tensor


@implements(Tensor, np.mean)
def mean(
    array: np.typing.ArrayLike, axis: Optional[int] = None, **kwargs: Any
) -> Tensor:
    if axis is None:
        den = float(np.prod(array.shape))
    else:
        den = array.shape[axis]

    return array.sum(axis=axis) / den


@implements(Tensor, np.max)
def npmax(*args: Any, **kwargs: Any) -> Tensor:
    args, kwargs = inputs2child(*args, **kwargs)  # type: ignore
    return np.max(*args, **kwargs)


@implements(Tensor, np.min)
def npmin(*args: Any, **kwargs: Any) -> Tensor:
    args, kwargs = inputs2child(*args, **kwargs)  # type: ignore
    return np.min(*args, **kwargs)


@implements(Tensor, np.square)
def square(x: np.typing.ArrayLike) -> Tensor:
    return x * x


@implements(Tensor, np.expand_dims)
def expand_dims(*args: Any, **kwargs: Any) -> Tensor:
    args, kwargs = inputs2child(*args, **kwargs)  # type: ignore
    return Tensor(np.expand_dims(*args, **kwargs))


@implements(Tensor, np.multiply)
def multiply(
    a: Union[Tensor, np.typing.ArrayLike], b: Union[Tensor, np.typing.ArrayLike]
) -> Tensor:
    if isinstance(a, Tensor):
        result = a.__mul__(b)
        if result is not NotImplementedError:
            return result

    if isinstance(b, Tensor):
        result = b.__rmul__(a)
        if result is not NotImplementedError:
            return result

    return TypeError(f"Can't multiply {type(a)} with {type(b)}")  # type: ignore
