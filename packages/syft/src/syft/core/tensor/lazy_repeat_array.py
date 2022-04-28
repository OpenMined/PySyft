# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import numpy as np

# relative
from ..common.serde.serializable import serializable
from .broadcastable import is_broadcastable
from .passthrough import is_acceptable_simple_type  # type: ignore
from .smpc.utils import get_shape


@serializable(recursive_serde=True)
class lazyrepeatarray:
    """
    When data is repeated along one or more dimensions, store it using lazyrepeatarray
    so that you can save on RAM and CPU when computing with it. Think like the opposite
    of np.broadcast, repeated values along an axis are collapsed but the .shape
    attribute of the higher dimensional projection is retained for operations.
    """

    __attr_allowlist__ = ["data", "shape"]

    def __init__(self, data: np.ndarray, shape: Tuple[int, ...]) -> None:
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

        self.data = data
        self.shape = shape

    def __add__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if is_acceptable_simple_type(other):
            return self.__class__(data=self.data + other, shape=self.shape)

        if not is_broadcastable(self.shape, other.shape):
            raise Exception(
                f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
            )

        if self.data.shape == other.data.shape:
            return self.__class__(data=self.data + other.data, shape=self.shape)
        else:
            # Rasswanth : Fix after addition of integration tests.
            print("Lazy Repeat adding with mismatched shapes")
            return self.__class__(data=self.data + other.data, shape=self.shape)

        raise Exception(f"not sure how to do this yet: {type(other)}")

    def __sub__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if is_acceptable_simple_type(other):
            return self.__class__(data=self.data - other, shape=self.shape)

        if self.shape != other.shape:
            raise Exception("cannot subtract tensors with different shapes")

        if self.data.shape == other.data.shape:
            return self.__class__(data=self.data - other.data, shape=self.shape)

        raise Exception("not sure how to do this yet")

    def __mul__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if is_acceptable_simple_type(other):
            return self.__class__(data=self.data * other, shape=self.shape)

        if self.shape != other.shape:
            raise Exception("cannot multiply tensors with different shapes")

        if self.data.shape == other.data.shape:
            return self.__class__(data=self.data * other.data, shape=self.shape)

        raise Exception("not sure how to do this yet")

    def __matmul__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if is_acceptable_simple_type(other):
            new_shape = get_shape("matmul", self.shape, other.shape)

            if self.data.size == 1:
                return self.__class__(
                    data=np.matmul(np.ones(self.shape), other * self.data),
                    shape=new_shape,
                )
            return self.__class__(data=self.data.__matmul__(other), shape=new_shape)

        if self.shape[-1] != other.shape[0]:
            raise Exception(
                "cannot matrix multiply tensors with different shapes: {self.shape} and {other.shape}"
            )

        result = self.to_numpy() @ other.to_numpy()
        return self.__class__(data=result, shape=result.shape)

        # raise Exception("not sure how to do this yet")

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

    def __eq__(self, other: Any) -> lazyrepeatarray:  # type: ignore
        if isinstance(other, lazyrepeatarray):
            if self.shape == other.shape:
                return lazyrepeatarray(data=self.data == other.data, shape=self.shape)
            else:
                result = (self.to_numpy() == other.to_numpy()).all()
                return lazyrepeatarray(data=np.array([result]), shape=result.shape)
        if isinstance(other, np.ndarray):
            try:
                _ = np.broadcast_shapes(self.shape, other.shape)
                result = (self.to_numpy() == other).all()
                return lazyrepeatarray(data=np.array([result]), shape=other.shape)
            except Exception as e:
                print(
                    "Failed to compare lazyrepeatarray with "
                    + f"{self.shape} == {other.shape} to numpy by broadcasting. {e}"
                )
                raise e

        return self == other

    def __le__(self, other: Any) -> lazyrepeatarray:  # type: ignore
        if isinstance(other, lazyrepeatarray):
            if self.shape == other.shape:
                return lazyrepeatarray(data=self.data <= other.data, shape=self.shape)
            else:
                result = (self.to_numpy() <= other.to_numpy()).all()
                return lazyrepeatarray(data=np.array([result]), shape=result.shape)
        if isinstance(other, np.ndarray):
            try:
                _ = np.broadcast_shapes(self.shape, other.shape)
                result = (self.to_numpy() <= other).all()
                return lazyrepeatarray(data=np.array([result]), shape=other.shape)
            except Exception as e:
                print(
                    "Failed to compare lazyrepeatarray with "
                    + f"{self.shape} == {other.shape} to numpy by broadcasting. {e}"
                )
                raise e

        return self <= other

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

    def to_numpy(self) -> np.ndarray:
        return np.broadcast_to(self.data, self.shape)

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
