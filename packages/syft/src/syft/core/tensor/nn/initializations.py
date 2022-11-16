# stdlib
from typing import Tuple

# third party
import numpy as np
from numpy.typing import NDArray

# relative
from ...common.serde.serializable import serializable


@serializable(recursive_serde=True)
class Initializer(object):
    __attr_allowlist__: Tuple[str] = ()  # type: ignore
    """Base class for parameter weight initializers.

    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.
    """

    def __call__(self, size: Tuple[int, ...]) -> NDArray:
        """Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`call()` method.
        """
        return self.call(size)

    def call(self, size: Tuple[int, ...]) -> NDArray:
        """Sample should return a numpy.array of size shape and data type
        ``numpy.float32``.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__


@serializable(recursive_serde=True)
class Uniform(Initializer):
    __attr_allowlist__ = ("scale",)

    def __init__(self, scale: float = 0.05):
        self.scale = scale

    def call(self, size: Tuple[int, ...]) -> NDArray:
        return np.array(np.random.uniform(-self.scale, self.scale, size=size))


def decompose_size(size: Tuple[int, ...]) -> Tuple[int, int]:
    if len(size) == 2:
        fan_in = size[0]
        fan_out = size[1]

    elif len(size) == 4 or len(size) == 5:
        respective_field_size = np.prod(size[2:])
        fan_in = size[1] * respective_field_size
        fan_out = size[0] * respective_field_size

    else:
        fan_in = fan_out = int(np.sqrt(np.prod(size)))

    return fan_in, fan_out


@serializable(recursive_serde=True)
class XavierInitialization(Initializer):
    __attr_allowlist__: Tuple[str] = ()  # type: ignore

    def call(self, size: Tuple[int, ...]) -> NDArray:
        fan_in, fan_out = decompose_size(size)
        return Uniform(np.sqrt(6 / (fan_in + fan_out)))(size)
