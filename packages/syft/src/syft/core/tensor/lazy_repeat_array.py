# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from xmlrpc.client import Boolean

# third party
import numpy as np

# relative
from ..common.serde.serializable import serializable
from .broadcastable import is_broadcastable


@serializable(recursive_serde=True)
class lazyrepeatarray:
    """
    When data is repeated along one or more dimensions, store it using lazyrepeatarray
    so that you can save on RAM and CPU when computing with it. Think like the opposite
    of np.broadcast, repeated values along an axis are collapsed but the .shape
    attribute of the higher dimensional projection is retained for operations.
    """

    __attr_allowlist__ = ["data", "shape"]

    def __init__(
        self,
        data: np.ndarray,
        shape: Tuple[int, ...],
        transforms: Optional[List] = None,
    ) -> None:
        """
        data: the raw data values without repeats
        shape: the shape of 'data' if repeats were included
        """

        # NOTE: all additional arguments are assumed to be broadcast if dims are shorter
        # than that of data. Example: if data.shape == (2,3,4) and
        # min_vals.shape == (2,3), then it's assumed that the full min_vals.shape is
        # actually (2,3,4) where the last dim is simply copied.
        # Example2: if data.shape == (2,3,4) and min_vals.shape == (2,1,4), then the
        # middle dimension is supposed to be copied to be min_vals.shape == (2,3,4)
        # if necessary. This is just to keep the memory footprint (and computation)
        # as small as possible.

        if isinstance(data, (bool, int, float)):
            data = np.array(data)

        # verify broadcasting works on shapes
        np.broadcast_shapes(data.shape, shape)

        if transforms is None:
            self.transforms = []
        else:
            self.transforms = transforms

        self.data = data
        self.shape = shape
        self._shape = self.shape

    def __add__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data + other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.add, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.add, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __sub__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data - other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.subtract, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.subtract, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __mul__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data * other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.multiply, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.multiply, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __matmul__(self, other: Any):  # type: ignore
        if isinstance(other, (int, np.integer, float, np.floating)):
            raise Exception

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if len(self.shape) != 2 or len(other.shape) != 2:
                raise Exception("Matmul only valid for 2D arrays")

            if self.shape[-1] != other.shape[0]:
                raise Exception(
                    "Cannot matrix multiply tensors with different shapes: {self.shape} and {other.shape}"
                )
            else:
                self.shape = (self.shape[0], other.shape[-1])
                if isinstance(other, lazyrepeatarray):
                    self.add_op(function=np.matmul, args=other.to_numpy())
                elif isinstance(other, np.ndarray):
                    self.add_op(function=np.matmul, args=other)
                return self

    def __eq__(self, other: Any) -> lazyrepeatarray:  # type: ignore
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data == other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.equal, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.equal, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __ne__(self, other: Any) -> lazyrepeatarray:  # type: ignore
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data != other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.not_equal, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.not_equal, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __le__(self, other: Any) -> lazyrepeatarray:
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data <= other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.less_equal, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.less_equal, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __ge__(self, other: Any) -> lazyrepeatarray:
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data >= other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.greater_equal, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.greater_equal, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __lt__(self, other: Any) -> lazyrepeatarray:
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data < other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.less, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.less, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __gt__(self, other: Any) -> lazyrepeatarray:
        if isinstance(other, (int, np.integer, float, np.floating)):
            return lazyrepeatarray(data=(self.data > other), shape=self.shape)

        elif isinstance(other, (np.ndarray, lazyrepeatarray)):  # type: ignore
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
                )
            self.shape = np.broadcast_shapes(self.shape, other.shape)
            if isinstance(other, lazyrepeatarray):
                self.add_op(function=np.greater, args=other.data)
            elif isinstance(other, np.ndarray):
                self.add_op(function=np.greater, args=other)
            return self

        else:
            raise Exception(f"not sure how to do this yet: {type(other)}")

    def __neg__(self) -> lazyrepeatarray:
        self.data *= -1
        return self

    def __pow__(self, exponent: int) -> lazyrepeatarray:
        if exponent == 2:
            return self * self
        raise Exception("not sure how to do this yet")

    def copy(self, order: Optional[str] = "K") -> lazyrepeatarray:
        return self.__class__(data=self.data.copy(order=order), shape=self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    def sum(self, *args: Tuple[Any, ...], **kwargs: Any) -> np.ndarray:
        if "axis" in kwargs and kwargs["axis"] is None:
            # TODO: make fast
            return self.to_numpy().sum()
        else:
            raise Exception("not sure how to do this yet")

    def concatenate(
        self, other: lazyrepeatarray, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> lazyrepeatarray:
        if not isinstance(other, lazyrepeatarray):
            raise NotImplementedError

        dummy_res = np.concatenate(
            (np.empty(self.shape), np.empty(other.shape)), *args, **kwargs
        )
        return lazyrepeatarray(data=self.data, shape=dummy_res.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def astype(self, np_type: np.dtype) -> lazyrepeatarray:
        return self.__class__(self.data.astype(np_type), self.shape)

    def to_numpy(self, original: Boolean = False) -> np.ndarray:
        if not original:
            return np.ones(self.shape) * self.data
        else:
            return np.ones(self._shape) * self.data

    def __repr__(self) -> str:
        return f"<lazyrepeatarray data: {self.data} -> shape: {self.shape}>"

    def __bool__(self) -> bool:
        return self.data.__bool__()

    def all(self) -> bool:
        return self.data.all()

    def any(self) -> bool:
        return self.data.any()

    def transpose(self, *args: List[Any], **kwargs: Dict[str, Any]) -> lazyrepeatarray:
        dummy_res = self.to_numpy().transpose(*args, **kwargs)
        return lazyrepeatarray(
            data=self.data.transpose(*args, **kwargs), shape=dummy_res.shape
        )

    def add_op(self, function: Callable, selection=slice(None), args=None) -> None:  # type: ignore
        self.transforms.append((function, selection, args))

    def evaluate(self):  # type: ignore
        result = self.to_numpy(original=True)
        for func, selection, args in self.transforms:
            if func == np.matmul:
                result = func(result[selection], args)
            else:
                result[selection] = func(result[selection], args)
        self.transforms = []
        return result
