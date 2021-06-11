# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
from typing import Tuple as TypeTuple
from typing import Union

# third party
import numpy as np

# syft relative
from .util import implements
from .util import query_implementation
from ..common.uid import UID

AcceptableSimpleType = Union[int, bool, float, np.ndarray]

def inputs2child(*args, **kwargs):
    args = [x.child if isinstance(x, PassthroughTensor) else x for x in args]
    kwargs = {
        x[0]: x[1].child if isinstance(x[1], PassthroughTensor) else x[1]
        for x in kwargs.items()
    }
    return args, kwargs


def is_acceptable_simple_type(obj):
    return isinstance(obj, (int, bool, float, np.ndarray))


class PassthroughTensor(np.lib.mixins.NDArrayOperatorsMixin):
    """A simple tensor class which passes method/function calls to self.child"""

    def __init__(self, child, id=None) -> None:
        self.child = child
        if id is None:
            self.id = UID() 
        else:
            self.id = id

    # TODO: Remove
    @property
    def _data_child(self) -> Any:
        data = self
        while hasattr(data, "child"):
            data = data.child
        return data

    def __len__(self) -> int:
        return len(self.child)

    @property
    def shape(self):
        return tuple(self.child.shape)

    def __abs__(self) -> PassthroughTensor:
        return self.__class__(self.child.__abs__())

    def __add__(self, other) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child + other)
        return self.__class__(self.child + other.child)

    def __radd__(self, other) -> PassthroughTensor:
        return other.__class__(other.child + self.child)

    def __sub__(self, other) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child - other)
        return self.__class__(self.child - other.child)

    def __rsub__(self, other) -> PassthroughTensor:
        return self.__class__(-((self - other).child))

    def __gt__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child > other)

        return self.__class__(self.child > other.child)

    def __ge__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child >= other)

        return self.__class__(self.child >= other.child)

    def __lt__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child < other)

        return self.__class__(self.child < other.child)

    def __le__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child <= other)

        return self.__class__(self.child <= other.child)

    def __ne__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child != other)

        return self.__class__(self.child != other.child)

    def __eq__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child == other)

        return self.__class__(self.child == other.child)

    def __floordiv__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__floordiv__(other))

        return self.__class__(self.child.__floordiv__(other.child))

    def __rfloordiv__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(other.__floordiv__(self.child))

        return self.__class__(other.child.__floordiv__(self.child))

    def __lshift__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__lshift__(other))

        return self.__class__(self.child.__lshift__(other.child))

    def __rlshift__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(other.__lshift__(self.child))

        return self.__class__(other.child.__lshift__(self.child))

    def __rshift__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__rshift__(other))

        return self.__class__(self.child.__rshift__(other.child))

    def __rrshift__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(other.__rshift__(self.child))

        return self.__class__(other.child.__rshift__(self.child))

    def __pow__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__pow__(other))

        return self.__class__(self.child.__pow__(other.child))

    def __rpow__(self, other: Union[PassthroughTensor, AcceptableSimpleType]):
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__rpow__(other))
        return self.__class__(self.child.__rpow__(other.child))

    def __divmod__(
        self,
        other: Union[PassthroughTensor, AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__divmod__(other))

        return self.__class__(self.child.__divmod__(other.child))

    def __neg__(self) -> PassthroughTensor:
        return self * -1

    def __index__(self) -> int:
        return self.child.__index__()

    def __invert__(self) -> PassthroughTensor:
        return self.child.__invert__()

    def copy(self) -> PassthroughTensor:
        return self.__class__(self.child.copy())

    def __mul__(self, other) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child * other)

        return self.__class__(self.child * other.child)

    def __rmul__(self, other) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__rmul__(other))
        return self.__class__(self.child.__rmul__(other.child))

    def __matmul__(
        self, other: Union[PassthroughTensor, np.ndarray]
    ) -> PassthroughTensor:
        return self.manual_dot(other)

    def __rmatmul__(
        self, other: Union[PassthroughTensor, np.ndarray]
    ) -> PassthroughTensor:
        return other.manual_dot(self)

    def __truediv__(self, other) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child * (1 / other))

        return self.__class__(self.child / other.child)

    def __rtruediv__(self, other) -> PassthroughTensor:
        return other.__truediv__(self)

    def manual_dot(self, other) -> PassthroughTensor:

        expanded_self = self.repeat(other.shape[1], axis=1)
        expanded_self = expanded_self.reshape(
            self.shape[0], self.shape[1], other.shape[1]
        )
        expanded_other = other.reshape([1] + list(other.shape)).repeat(
            self.shape[0], axis=0
        )

        prod = expanded_self * expanded_other
        result = prod.sum(axis=1)
        return result

    def dot(self, other) -> PassthroughTensor:
        return self.manual_dot(other)
        # if isinstance(other, self.__class__):
        #     return self.__class__(self.child.dot(other.child))
        # return self.__class__(self.child.dot(other))

    def reshape(self, *dims) -> PassthroughTensor:
        return self.__class__(self.child.reshape(*dims))

    def repeat(self, *args, **kwargs) -> PassthroughTensor:
        return self.__class__(self.child.repeat(*args, **kwargs))

    # TODO: why does this version of repeat fail but the *args **kwargs one works?
    # def repeat(
    #     self, repeats: Union[int, TypeTuple[int, ...]], axis: Optional[int] = None
    # ) -> PassthroughTensor:
    #     return self.__class__(self.child.repeat(repeats, axis=axis))

    # numpy.resize(a, new_shape)
    def resize(self, new_shape: Union[int, TypeTuple[int, ...]]) -> PassthroughTensor:
        return self.__class__(self.child.resize(new_shape))

    # def sum(self, dim):
    #     return self.__class__(self.child.sum(dim))

    @property
    def T(self) -> PassthroughTensor:
        return self.transpose()

    def transpose(self, *args, **kwargs):
        return self.__class__(self.child.transpose(*args, **kwargs))

    def __getitem__(
        self, key: Union[int, bool, np.array, PassthroughTensor, slice, Ellipsis]
    ) -> Union[PassthroughTensor, AcceptableSimpleType]:
        if isinstance(key, PassthroughTensor):
            return self.__class__(self.child.__getitem__(key.child))

        return self.__class__(self.child.__getitem__(key))

    # numpy.argmax(a, axis=None, out=None)
    def argmax(self, axis: Optional[int]) -> PassthroughTensor:
        return self.__class__(self.child.argmax(axis))

    def argmin(self, axis: Optional[int]) -> PassthroughTensor:
        return self.__class__(self.child.argmin(axis))

    # numpy.argsort(a, axis=-1, kind=None, order=None)
    def argsort(self, axis: Optional[int] = -1) -> PassthroughTensor:
        return self.__class__(self.child.argsort(axis))

    # ndarray.sort(axis=-1, kind=None, order=None)
    def sort(self, axis: int = -1, kind: Optional[str] = None) -> PassthroughTensor:
        return self.__class__(self.child.sort(axis=axis, kind=kind))

    # numpy.clip(a, a_min, a_max, out=None, **kwargs)
    def clip(
        self,
        min: Optional[AcceptableSimpleType] = None,
        max: Optional[AcceptableSimpleType] = None,
    ) -> PassthroughTensor:
        return self.__class__(self.child.clip(min, max))

    # numpy.cumprod(a, axis=None, dtype=None, out=None)
    def cumprod(self, axis: Optional[int] = None) -> PassthroughTensor:
        return self.__class__(self.child.cumprod(axis=axis))

    # numpy.cumsum(a, axis=None, dtype=None, out=None)
    def cumsum(self, axis: Optional[int] = None) -> PassthroughTensor:
        return self.__class__(self.child.cumsum(axis=axis))

    # numpy.diagonal(a, offset=0, axis1=0, axis2=1)
    def diagonal(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> PassthroughTensor:
        return self.__class__(self.child.diagonal(offset, axis1, axis2))

    def tolist(self) -> PassthroughTensor:
        return self.__class__(self.child.tolist())

    # ndarray.flatten(order='C')
    def flatten(self, order: str = "C") -> PassthroughTensor:
        return self.__class__(self.child.flatten(order))

    # numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)
    def mean(
        self, axis: Optional[Union[int, TypeTuple[int, ...]]] = None, **kwargs
    ) -> Union[PassthroughTensor, np.floating]:
        return self.__class__(self.child.mean(axis=axis, **kwargs))

    # ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
    def max(
        self, axis: Optional[Union[int, TypeTuple[int, ...]]] = None
    ) -> PassthroughTensor:
        return self.__class__(self.child.max(axis=axis))

    # ndarray.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
    def min(
        self, axis: Optional[Union[int, TypeTuple[int, ...]]] = None
    ) -> PassthroughTensor:
        return self.__class__(self.child.min(axis=axis))

    @property
    def ndim(self) -> np.unsignedinteger:
        return self.child.ndim

    # ndarray.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)
    def prod(
        self, axis: Optional[Union[int, TypeTuple[int, ...]]] = None
    ) -> PassthroughTensor:
        return self.__class__(self.child.prod(axis=axis))

    # numpy.squeeze(a, axis=None)
    def squeeze(
        self, axis: Optional[Union[int, TypeTuple[int, ...]]] = None
    ) -> PassthroughTensor:
        return self.__class__(self.child.squeeze(axis=axis))

    # numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)
    def std(
        self, axis: Optional[Union[int, TypeTuple[int, ...]]] = None
    ) -> PassthroughTensor:
        return self.__class__(self.child.std(axis=axis))

    # numpy.sum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
    def sum(
        self, axis: Optional[Union[int, TypeTuple[int, ...]]] = None
    ) -> PassthroughTensor:
        result = self.child.sum(axis=axis)
        return self.__class__(result)

    # numpy.take(a, indices, axis=None, out=None, mode='raise')
    def take(
        self, indices: Optional[Union[int, TypeTuple[int, ...]]] = None
    ) -> PassthroughTensor:
        return self.__class__(self.child.take(indices=indices))

    def __array_function__(self, func, types, args, kwargs):
        # Note: this allows subclasses that don't override
        # __array_function__ to handle PassthroughTensor objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        implementation = query_implementation(self.__class__, func)
        if implementation:
            return implementation(*args, **kwargs)
        return self.__class__(func(*args, **kwargs))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        implementation = query_implementation(self.__class__, ufunc)
        if implementation:
            return implementation(*inputs, **kwargs)
        return self.__class__(ufunc(*inputs, **kwargs))

    def __repr__(self):
        return f"{self.__class__.__name__}(child={self.child})"


@implements(PassthroughTensor, np.square)
def square(x):
    return x * x
