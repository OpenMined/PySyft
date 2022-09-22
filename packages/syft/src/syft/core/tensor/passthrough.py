# type: ignore
# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple as TypeTuple
from typing import Type
from typing import Union

# third party
import jaxlib
import numpy as np
import torch

# relative
from .util import implements
from .util import query_implementation

AcceptableSimpleType = Union[int, bool, float, np.ndarray]
SupportedChainType = Union["PassthroughTensor", AcceptableSimpleType]


def is_acceptable_simple_type(obj):
    return isinstance(
        obj,
        (
            int,
            bool,
            float,
            np.ndarray,
            torch.Tensor,
            jaxlib.xla_extension.DeviceArrayBase,
        ),
    )


class PassthroughTensor(np.lib.mixins.NDArrayOperatorsMixin):
    """A simple tensor class which passes method/function calls to self.child"""

    def __init__(self, child: Any) -> None:
        self.child = child

    # TODO: Remove
    @property
    def _data_child(self) -> Any:
        data = self
        while hasattr(data, "child"):
            data = data.child
        return data

    def __len__(self) -> int:
        if (
            isinstance(self.child, float)
            or isinstance(self.child, int)
            or isinstance(self.child, bool)
        ):
            return 1
        return len(self.child)

    @property
    def shape(self) -> TypeTuple[Any, ...]:
        if (
            isinstance(self.child, float)
            or isinstance(self.child, int)
            or isinstance(self.child, bool)
        ):
            return (1,)
        return tuple(self.child.shape)

    @property
    def size(self) -> int:
        if (
            isinstance(self.child, float)
            or isinstance(self.child, int)
            or isinstance(self.child, bool)
        ):
            return 1

        if hasattr(self.child, "size"):
            return self.child.size
        elif hasattr(self.child, "shape"):
            return np.prod(self.child.shape)

        raise Exception(f"{type(self)} has no attribute size.")

    @property
    def dtype(self) -> np.dtype:
        return self.child.dtype

    def zeros_like(self) -> PassthroughTensor:
        if is_acceptable_simple_type(self.child):
            return np.zeros_like(self.child)
        return self.child.zeros_like()

    # @property
    # def shape(self) -> Union[TypeTuple[Any, ...], List[Any]]:
    #     """There are 3 options for where shape information can be sourced from:

    #     - self.client_shape: which is a logical attempt on the client side to infer
    #       shape information
    #     - self.child.shape: which just says that this layer of logical abstraction isn't
    #       responsible for tracking shape and inhereits it from the child
    #     - Pointer.shape: a special case of the other two wherein shape is called on a
    #       Pointer and it fetches that information from a remote object if
    #       Pointer.client_shape is not available (is None).

    #     By default the priority for where shape information should come from is:
    #     - self.client_shape is first because this doesn't leverage private information.
    #     - self.child.shape is second as a delegation of the first
    #     - Pointer.client_shape as a special case of the previous
    #     - Pointer.remote_shape is a last resort because it requires calling .request()
    #       and .get()

    #     """
    #     if self.client_shape is not None:
    #         return self.client_shape

    #     elif self.child is not None:

    #         # note that this will attempt client_shape on the child before
    #         # anything else because it too will call this method
    #         shape = self.child.shape

    #         if isinstance(shape, Pointer):
    #             return shape.get(request_block=True)
    #         else:
    #             return shape

    #     else:
    #         msg = (
    #             "Not sure how to find shape because self.client_shape and self.child"
    #             + "are both none"
    #         )
    #         raise Exception(msg)

    #     return tuple(self.child.shape)

    def __and__(self, other):
        if is_acceptable_simple_type(other):
            return self.__class__(self.child & other)
        return self.__class__(self.child & other.child)

    def __rand__(self, other):
        if is_acceptable_simple_type(other):
            return self.__class__(other & self.child)

        return other.__class__(other.child & self.child)

    def __abs__(self) -> Union[Type[PassthroughTensor], AcceptableSimpleType]:
        return self.__class__(self.child.__abs__())

    def __add__(
        self, other: Union[Type[PassthroughTensor], AcceptableSimpleType]
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child + other)
        return self.__class__(self.child + other.child)

    def __radd__(self, other) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(other + self.child)

        return other.__class__(other.child + self.child)

    def __sub__(
        self, other: Union[Type[PassthroughTensor], AcceptableSimpleType]
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child - other)
        return self.__class__(self.child - other.child)

    def __rsub__(self, other) -> PassthroughTensor:
        return self.__class__(-(self - other).child)

    def __gt__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child > other)

        return self.__class__(self.child > other.child)

    def __ge__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child >= other)

        return self.__class__(self.child >= other.child)

    def __lt__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child < other)

        return self.__class__(self.child < other.child)

    def __le__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child <= other)

        return self.__class__(self.child <= other.child)

    def __ne__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child != other)

        return self.__class__(self.child != other.child)

    def __eq__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child == other)

        return self.__class__(self.child == other.child)

    def __floordiv__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__floordiv__(other))

        return self.__class__(self.child.__floordiv__(other.child))

    def __rfloordiv__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(other.__floordiv__(self.child))

        return self.__class__(other.child.__floordiv__(self.child))

    def __lshift__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__lshift__(other))

        return self.__class__(self.child.__lshift__(other.child))

    def __rlshift__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(other.__lshift__(self.child))

        return self.__class__(other.child.__lshift__(self.child))

    def __rshift__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__rshift__(other))

        return self.__class__(self.child.__rshift__(other.child))

    def __rrshift__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(other.__rshift__(self.child))

        return self.__class__(other.child.__rshift__(self.child))

    def __pow__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__pow__(other))

        return self.__class__(self.child.__pow__(other.child))

    def __rpow__(
        self, other: Union[Type[PassthroughTensor], AcceptableSimpleType]
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__rpow__(other))
        return self.__class__(self.child.__rpow__(other.child))

    def __divmod__(
        self,
        other: Union[Type[PassthroughTensor], AcceptableSimpleType],
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

    def copy(self, order: Optional[str] = "K") -> PassthroughTensor:
        return self.__class__(self.child.copy(order=order))

    def __mul__(
        self, other: Union[Type[PassthroughTensor], AcceptableSimpleType]
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child * other)

        return self.__class__(self.child * other.child)

    def concatenate(
        self, other: PassthroughTensor, *args, **kwargs
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            raise ValueError("Does not currently for Simple Types")

        return self.__class__(self.child.concatenate(other.child, *args, **kwargs))

    def __rmul__(
        self, other: Union[Type[PassthroughTensor], AcceptableSimpleType]
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__rmul__(other))
        return self.__class__(self.child.__rmul__(other.child))

    def __matmul__(
        self, other: Union[Type[PassthroughTensor], np.ndarray]
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child @ other)

        return self.__class__(self.child @ other.child)

    def __rmatmul__(
        self, other: Union[Type[PassthroughTensor], np.ndarray]
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child.__rmatmul__(other))

        return self.__class__(self.child.__rmatmul__(other.child))

    def __truediv__(
        self, other: Union[Type[PassthroughTensor], AcceptableSimpleType]
    ) -> PassthroughTensor:
        if is_acceptable_simple_type(other):
            return self.__class__(self.child * (1 / other))  # type: ignore

        return self.__class__(self.child / other.child)

    def __rtruediv__(self, other: Type[PassthroughTensor]) -> PassthroughTensor:
        return other.__truediv__(self)

    def manual_dot(
        self, other: Union[Type[PassthroughTensor], np.ndarray]
    ) -> PassthroughTensor:

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

    def dot(
        self, other: Union[Type[PassthroughTensor], np.ndarray]
    ) -> PassthroughTensor:
        return self.manual_dot(other)

    def reshape(self, *dims) -> PassthroughTensor:
        return self.__class__(self.child.reshape(*dims))

    def repeat(
        self, repeats: Union[int, TypeTuple[int, ...]], axis: Optional[int] = None
    ) -> PassthroughTensor:
        return self.__class__(self.child.repeat(repeats, axis))

    def resize(
        self,
        new_shape: Union[int, TypeTuple[int, ...]],
        refcheck: Optional[bool] = True,
    ) -> PassthroughTensor:
        # Should be modified to  remove copy
        # https://stackoverflow.com/questions/23253144/numpy-the-array-doesnt-have-its-own-data
        res = self.child.copy()
        res.resize(new_shape, refcheck=refcheck)
        return self.__class__(res)

    @property
    def T(self) -> PassthroughTensor:
        return self.transpose()

    def transpose(self, *args, **kwargs) -> PassthroughTensor:
        return self.__class__(self.child.transpose(*args, **kwargs))

    def __getitem__(
        self, key: Union[int, bool, np.array, PassthroughTensor, slice]
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
    def cumsum(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
    ) -> PassthroughTensor:
        return self.__class__(self.child.cumsum(axis=axis, dtype=dtype, out=out))

    # numpy.trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None)
    def trace(
        self,
        offset: Optional[int] = 0,
        axis1: Optional[int] = 0,
        axis2: Optional[int] = 1,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
    ) -> PassthroughTensor:
        return self.__class__(
            self.child.trace(
                offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
            )
        )

    # numpy.diagonal(a, offset=0, axis1=0, axis2=1)
    def diagonal(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> PassthroughTensor:
        return self.__class__(self.child.diagonal(offset, axis1, axis2))

    def tolist(self) -> PassthroughTensor:
        return self.__class__(self.child.tolist())

    # ndarray.flatten(order='C')
    def flatten(self, order: Optional[str] = "C") -> PassthroughTensor:
        return self.__class__(self.child.flatten(order))

    # ndarray.partition(kth, axis=- 1, kind='introselect', order=None)
    def partition(
        self,
        kth: Union[int, TypeTuple[int, ...]],
        axis: Optional[int] = -1,
        kind: Optional[str] = "introselect",
        order: Optional[Union[int, TypeTuple[int, ...]]] = None,
    ) -> PassthroughTensor:
        self.child.partition(kth=kth, axis=axis, kind=kind, order=order)  # inplace op
        return self.__class__(self.child)

    # ndarray.ravel([order])
    def ravel(self, order: Optional[str] = "C") -> PassthroughTensor:
        return self.__class__(self.child.ravel(order=order))

    # ndarray.compress(condition, axis=None, out=None)
    def compress(
        self, condition: List[bool], axis: int = None, out: Optional[np.ndarray] = None
    ) -> PassthroughTensor:
        return self.__class__(
            self.child.compress(condition=condition, axis=axis, out=out)
        )

    # ndarray.swapaxes(axis1, axis2)
    def swapaxes(self, axis1: int, axis2: int) -> PassthroughTensor:
        return self.__class__(self.child.swapaxes(axis1, axis2))

    # ndarray.put(indices, values, mode='raise')
    def put(
        self,
        indices: Union[int, TypeTuple[int, ...], np.ndarray],
        values: Union[int, TypeTuple[int, ...], np.ndarray],
        mode: Optional[str] = "raise",
    ) -> None:
        self.child.put(indices=indices, values=values, mode=mode)  # inplace op
        print(self.__class__)
        print(self.child)
        return self.__class__(self.child)

    # ndarray.__pos__(/)
    def __pos__(self) -> PassthroughTensor:
        return self.__class__(self.child.__pos__())

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

    def sum(self, *args, **kwargs) -> PassthroughTensor:
        result = self.child.sum(*args, **kwargs)
        if hasattr(self, "copy_tensor"):
            tensor = self.copy_tensor()
            tensor.child = result
            return tensor
        return self.__class__(result)

    def ones_like(self, *args, **kwargs) -> PassthroughTensor:
        result = self.child.ones_like(*args, **kwargs)

        return self.__class__(result)

    # numpy.take(a, indices, axis=None, out=None, mode='raise')
    def take(
        self,
        indices: Union[int, TypeTuple[int, ...], np.ndarray],
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        mode: Optional[str] = "raise",
    ) -> PassthroughTensor:
        return self.__class__(
            self.child.take(
                indices,
                axis=axis,
                out=out,
                mode=mode,
            )
        )

    # numpy.choose(a, choices, out=None, mode='raise')
    def choose(
        self,
        choices: Sequence[Union[PassthroughTensor, np.ndarray]],
        out: Optional[np.ndarray] = None,
        mode: Optional[str] = "raise",
    ) -> PassthroughTensor:
        return self.__class__(
            self.child.choose(
                choices,
                out=out,
                mode=mode,
            )
        )

    def decode(self) -> AcceptableSimpleType:
        return self.child.decode()

    def astype(self, np_type: np.dtype) -> PassthroughTensor:
        return self.__class__(self.child.astype(np_type))

    def __array_function__(
        self,
        func: Callable,
        types: List[Type],  # what this means =  List of Type(Type())
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> PassthroughTensor:
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
def square(x: Type[PassthroughTensor]) -> PassthroughTensor:
    return x * x
