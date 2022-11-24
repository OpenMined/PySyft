# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union

# third party
import numpy as np
from scipy.ndimage.interpolation import rotate

# relative
from ..common.serde.serializable import serializable
from .broadcastable import is_broadcastable
from .config import DEFAULT_FLOAT_NUMPY_TYPE
from .config import DEFAULT_INT_NUMPY_TYPE
from .passthrough import is_acceptable_simple_type  # type: ignore
from .smpc.utils import get_shape

if TYPE_CHECKING:
    # relative
    from .autodp.phi_tensor import PhiTensor


@serializable(recursive_serde=True)
class lazyrepeatarray:
    """
    A class representing Differential Privacy metadata (minimum and maximum values) in a way that saves RAM/CPU.

    We store large arrays of a single repeating value as a single tuple (shape) and a single value (int/float/etc)
    e.g. np.array([8,8,8,8,8,8]) = lazyrepeatarray(data=8, shape=(6,))

    Think like the opposite of np.broadcast, repeated values along an axis are collapsed but the .shape
    attribute of the higher dimensional projection is retained for operations.

    ...

    Attributes:
        data: int/float
            the actual value that is repeating.
        shape: tuple
            the shape that the fully expanded array would be.

    Methods:
        to_numpy():
            expands the lazyrepeatarray into the full sized numpy array it was representing.

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
            if isinstance(data, int):
                data = data.astype(DEFAULT_INT_NUMPY_TYPE)  # type: ignore
            if isinstance(data, float):
                data = data.astype(DEFAULT_FLOAT_NUMPY_TYPE)  # type: ignore

        # verify broadcasting works on shapes
        if -1 not in shape:
            np.broadcast_shapes(data.shape, shape)

        self.data = data
        self.shape = shape
        if isinstance(shape, Iterable):
            for val in shape:
                if val < 0:
                    raise ValueError(f"Invalid shape: {shape}")

    def __getitem__(self, item: Union[str, int, slice]) -> lazyrepeatarray:
        if self.data.shape == self.shape:
            output = self.data[item]
            return lazyrepeatarray(data=output, shape=output.shape)
        elif self.data.size == 1:
            test_arr = np.ones(self.shape)[
                item
            ]  # TODO: Is there a better way to determine output shape?
            return lazyrepeatarray(data=self.data, shape=test_arr.shape)
        else:
            raise NotImplementedError

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
            return self.__class__(data=self.data + other.data, shape=self.shape)

    def __sub__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if is_acceptable_simple_type(other):
            res = self.data - other
            return self.__class__(data=res, shape=self.shape)

        if not is_broadcastable(self.shape, other.shape):
            raise Exception(
                f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
            )

        if self.data.shape == other.data.shape:
            return self.__class__(data=self.data - other.data, shape=self.shape)
        else:
            return self.__class__(data=self.data - other.data, shape=self.shape)

    def __mul__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if is_acceptable_simple_type(other):
            return self.__class__(data=self.data * other, shape=self.shape)

        if not is_broadcastable(self.shape, other.shape):
            raise Exception(
                "Cannot broadcast arrays with shapes for LazyRepeatArray Multiplication:"
                + f" {self.shape} & {other.shape}"
            )
        else:
            return self.__class__(data=self.data * other.data, shape=self.shape)

    def __matmul__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if is_acceptable_simple_type(other):
            new_shape = get_shape("__matmul__", self.shape, other.shape)

            if self.data.size == 1:
                return self.__class__(
                    data=np.matmul(np.ones(self.shape), other * self.data),
                    shape=new_shape,
                )
            return self.__class__(data=self.data.__matmul__(other), shape=new_shape)

        if self.shape[-1] != other.shape[-2]:
            raise Exception(
                f"cannot matrix multiply tensors with different shapes: {self.shape} and {other.shape}"
            )

        result = self.to_numpy() @ other.to_numpy()
        return self.__class__(data=result, shape=result.shape)

    def __lshift__(self, other: Any) -> lazyrepeatarray:
        if is_acceptable_simple_type(other):
            return self.__class__(data=self.data << other, shape=self.shape)

        if not is_broadcastable(self.shape, other.shape):
            raise Exception(
                f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
            )
        return self.__class__(data=self.data << other.data, shape=self.shape)

    def __rshift__(self, other: Any) -> lazyrepeatarray:
        if is_acceptable_simple_type(other):
            return self.__class__(data=self.data >> other, shape=self.shape)

        if not is_broadcastable(self.shape, other.shape):
            raise Exception(
                f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
            )
        return self.__class__(data=self.data >> other.data, shape=self.shape)

    def zeros_like(self, *args: Any, **kwargs: Any) -> lazyrepeatarray:
        res = np.array(np.zeros_like(self.to_numpy(), *args, **kwargs))
        return lazyrepeatarray(data=res, shape=res.shape)

    def __rtruediv__(self, other: Any) -> lazyrepeatarray:
        res = (1 / self.data) * other
        return lazyrepeatarray(data=res, shape=self.shape)

    def __truediv__(self, other: Any) -> lazyrepeatarray:
        if is_acceptable_simple_type(other):
            return self.__class__(data=self.data / other, shape=self.shape)

        if not is_broadcastable(self.shape, other.shape):
            raise Exception(
                "Cannot broadcast arrays with shapes for LazyRepeatArray FloorDiv:"
                + f" {self.shape} & {other.shape}"
            )
        else:
            return self.__class__(data=self.data / other.data, shape=self.shape)

    def __floordiv__(self, other: Any) -> lazyrepeatarray:
        if is_acceptable_simple_type(other):
            return self.__class__(data=self.data // other, shape=self.shape)

        if not is_broadcastable(self.shape, other.shape):
            raise Exception(
                "Cannot broadcast arrays with shapes for LazyRepeatArray FloorDiv:"
                + f" {self.shape} & {other.shape}"
            )
        else:
            return self.__class__(data=self.data // other.data, shape=self.shape)

    def __rmatmul__(self, other: Any) -> lazyrepeatarray:
        """
        THIS MIGHT LOOK LIKE COPY-PASTED CODE!
        Don't touch it. It's going to get more complicated.
        """
        if is_acceptable_simple_type(other):
            new_shape = get_shape("__matmul__", other.shape, self.shape)

            if other.size == 1:
                return self.__class__(
                    data=np.matmul(np.ones(other.shape), other * self.data),
                    shape=new_shape,
                )
            return self.__class__(
                data=self.to_numpy().__rmatmul__(other), shape=new_shape
            )

        if other.shape[-1] != self.shape[0]:
            raise Exception(
                "cannot matrix multiply tensors with different shapes: {self.shape} and {other.shape}"
            )

        result = self.to_numpy().__rmatmul__(other.to_numpy())
        return self.__class__(data=result, shape=result.shape)

    def __pow__(self, exponent: int) -> lazyrepeatarray:
        if exponent == 2:
            return self * self
        raise Exception("not sure how to do this yet")

    def pad(self, pad_width: int, mode: str = "reflect") -> lazyrepeatarray:
        if mode == "reflect":
            new_shape = tuple([i + pad_width * 2 for i in self.shape])
            if self.data.shape == self.shape:
                return lazyrepeatarray(
                    data=np.pad(self.data, pad_width=pad_width, mode="reflect"),
                    shape=new_shape,
                )
            elif self.data.size == 1:
                return lazyrepeatarray(data=self.data, shape=new_shape)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def horizontal_flip(self) -> lazyrepeatarray:
        if self.data.shape == self.shape:
            return lazyrepeatarray(data=np.fliplr(self.data), shape=self.shape)
        elif self.data.size == 1:
            return lazyrepeatarray(data=self.data, shape=self.shape)
        else:
            raise NotImplementedError

    def vertical_flip(self) -> lazyrepeatarray:
        if self.data.shape == self.shape:
            return lazyrepeatarray(data=np.flipud(self.data), shape=self.shape)
        elif self.data.size == 1:
            return lazyrepeatarray(data=self.data, shape=self.shape)
        else:
            raise NotImplementedError

    def rotate(self, angle: int) -> lazyrepeatarray:
        if self.data.shape == self.shape:
            return lazyrepeatarray(data=rotate(self.data, angle), shape=self.shape)
        elif self.data.size == 1:
            # TODO: This is almost certainly incorrect
            return lazyrepeatarray(data=self.data, shape=self.shape)
        else:
            raise NotImplementedError

    def reshape(self, target_shape: Tuple) -> lazyrepeatarray:
        # TODO: Can we reshape without creating new objects
        if self.data.shape == self.shape:
            return lazyrepeatarray(
                data=self.data.reshape(target_shape), shape=target_shape
            )
        elif self.data.size == 1:
            return lazyrepeatarray(data=self.data, shape=target_shape)
        else:
            if not np.broadcast_shapes(self.data.shape, target_shape):
                raise NotImplementedError(
                    f"data= {self.data.shape}, shape: {self.shape}"
                )
            else:
                return lazyrepeatarray(data=self.data, shape=target_shape)

    def copy(self, order: Optional[str] = "K") -> lazyrepeatarray:
        return self.__class__(data=self.data.copy(order=order), shape=self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    def sum(self, *args: Any, **kwargs: Any) -> lazyrepeatarray:
        res = np.array(self.to_numpy().sum(*args, **kwargs))
        return lazyrepeatarray(data=res, shape=res.shape)

    def ones_like(self, *args: Any, **kwargs: Any) -> lazyrepeatarray:
        res = np.array(np.ones_like(self.to_numpy(), *args, **kwargs))
        return lazyrepeatarray(data=res, shape=res.shape)

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
        self, other: lazyrepeatarray, *args: Any, **kwargs: Any
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

        # FIX: shape is not set sometimes
        if not self.shape:
            self.shape = self.data.shape

        return np.broadcast_to(self.data, self.shape)

    def __repr__(self) -> str:
        return f"<lazyrepeatarray data: {self.data} -> shape: {self.shape}>"

    def __bool__(self) -> bool:
        return self.data.__bool__()

    def all(self) -> bool:
        return self.data.all()

    def any(self) -> bool:
        return self.data.any()

    def transpose(self, *args: Any, **kwargs: Any) -> lazyrepeatarray:
        dummy_res = self.to_numpy().transpose(*args, **kwargs)
        return lazyrepeatarray(
            data=self.data.transpose(*args, **kwargs), shape=dummy_res.shape
        )

    def clip(self, *args: Any, **kwargs: Any) -> lazyrepeatarray:
        res = np.array(self.to_numpy().clip(*args, **kwargs))
        return lazyrepeatarray(data=res, shape=res.shape)


# As the min and max values calculation is the same regardless of the tensor type,
# We centralize this method as baseline for calculation for min/max values
def compute_min_max(
    x_min_vals: lazyrepeatarray,
    x_max_vals: lazyrepeatarray,
    other: Union[PhiTensor, int, float, np.ndarray],
    op_str: str,
    *args: Any,
    **kwargs: Dict[Any, Any],
) -> Tuple[lazyrepeatarray, lazyrepeatarray]:
    min_vals: lazyrepeatarray
    max_vals: lazyrepeatarray

    if op_str in [
        "__add__",
        "__matmul__",
        "__rmatmul__",
        "__truediv__",
        "__floordiv__",
    ]:
        if is_acceptable_simple_type(other):
            min_vals = getattr(x_min_vals, op_str)(other)
            max_vals = getattr(x_max_vals, op_str)(other)
        elif hasattr(other, "min_vals") and hasattr(other, "max_vals"):
            min_vals = getattr(x_min_vals, op_str)(other.min_vals)  # type: ignore
            max_vals = getattr(x_max_vals, op_str)(other.max_vals)  # type: ignore
        else:
            raise ValueError(
                f"Not supported type for lazy repeat array computation: {type(other)}"
            )

    elif op_str in ["__sub__", "__mul__"]:
        if is_acceptable_simple_type(other):
            min_vals = getattr(x_min_vals, op_str)(other)
            max_vals = getattr(x_max_vals, op_str)(other)
        elif hasattr(other, "min_vals") and hasattr(other, "max_vals"):
            min_min = getattr(x_min_vals.data, op_str)(other.min_vals.data)  # type: ignore
            min_max = getattr(x_min_vals.data, op_str)(other.max_vals.data)  # type: ignore
            max_min = getattr(x_max_vals.data, op_str)(other.min_vals.data)  # type: ignore
            max_max = getattr(x_max_vals.data, op_str)(other.max_vals.data)  # type: ignore
            _min_vals = np.minimum.reduce([min_min, min_max, max_min, max_max])
            _max_vals = np.maximum.reduce([min_min, min_max, max_min, max_max])
            min_vals = x_min_vals.copy()
            min_vals.data = _min_vals
            max_vals = x_max_vals.copy()
            max_vals.data = _max_vals
        else:
            raise ValueError(
                f"Not supported type for lazy repeat array computation: {type(other)}"
            )

    elif op_str in [
        "__gt__",
        "__lt__",
        "__le__",
        "__ge__",
        "__eq__",
        "__ne__",
    ]:
        min_vals = x_min_vals * 0
        max_vals = (x_max_vals * 0) + 1
    elif op_str == "sum":
        min_vals = x_min_vals.sum(*args, **kwargs)
        max_vals = x_max_vals.sum(*args, **kwargs)
    elif op_str in ["__pos__", "sort"]:
        min_vals = x_min_vals
        max_vals = x_max_vals
    elif op_str == "trace":
        # NOTE: This is potentially expensive
        min_val_data = x_min_vals.to_numpy().trace(*args, **kwargs)
        min_vals = lazyrepeatarray(data=min_val_data, shape=min_val_data.shape)
        max_val_data = x_max_vals.to_numpy().trace(*args, **kwargs)
        max_vals = lazyrepeatarray(data=max_val_data, shape=max_val_data.shape)
    elif op_str == "repeat":
        dummy_res = np.empty(x_min_vals.shape).repeat(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data.min(), shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data.max(), shape=dummy_res.shape)
    elif op_str == "diagonal":
        dummy_res = np.empty(x_min_vals.shape).diagonal(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data.min(), shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data.max(), shape=dummy_res.shape)
    elif op_str == "clip":
        dummy_res = np.empty(x_min_vals.shape).clip(*args, **kwargs)
        min_v = np.clip(x_min_vals.data, *args, **kwargs)
        max_v = np.clip(x_max_vals.data, *args, **kwargs)
        min_vals = lazyrepeatarray(data=min_v, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=max_v, shape=dummy_res.shape)
    elif op_str == "choose":
        # relative
        from .tensor import Pointer

        dummy_res = np.ones(x_min_vals.shape, dtype=np.int64)
        if isinstance(args[0], Pointer):
            temp_args = (np.ones(args[0].shape), *args[1:])  # type: ignore
            dummy_res = dummy_res.choose(*temp_args, **kwargs)
            min_vals = lazyrepeatarray(
                data=x_min_vals.data.min(), shape=dummy_res.shape
            )
            max_vals = lazyrepeatarray(
                data=x_max_vals.data.max(), shape=dummy_res.shape
            )
        else:
            dummy_res = dummy_res.choose(*args, **kwargs)
            min_vals = lazyrepeatarray(
                data=x_min_vals.data.min(), shape=dummy_res.shape
            )
            max_vals = lazyrepeatarray(
                data=x_max_vals.data.max(), shape=dummy_res.shape
            )
    elif op_str == "min":
        dummy_res = np.empty(x_min_vals.shape).min(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "max":
        dummy_res = np.empty(x_min_vals.shape).max(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "zeros_like":
        min_vals = x_min_vals.zeros_like(*args, **kwargs)
        max_vals = x_max_vals.zeros_like(*args, **kwargs)
    elif op_str == "ones_like":
        min_vals = x_min_vals.ones_like(*args, **kwargs)
        max_vals = x_max_vals.ones_like(*args, **kwargs)
    elif op_str == "copy":
        min_vals = x_min_vals.copy(*args, **kwargs)  # type: ignore
        max_vals = x_max_vals.copy(*args, **kwargs)  # type: ignore
    elif op_str == "mean":
        dummy_res = np.empty(x_min_vals.shape).mean(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "round":
        min_vals = lazyrepeatarray(
            data=x_min_vals.data.round(*args, **kwargs), shape=x_min_vals.shape
        )
        max_vals = lazyrepeatarray(
            data=x_max_vals.data.round(*args, **kwargs), shape=x_max_vals.shape
        )
    elif op_str == "__pow__":
        if x_min_vals.data <= 0 <= x_max_vals.data:
            # If data is in range [-5, 5], it's possible the minimum is 0 and not (-5)^2
            min_data = min(0, (x_min_vals.data.__pow__(*args, **kwargs)).min())
        else:
            min_data = x_min_vals.data.__pow__(*args, **kwargs)
        min_vals = lazyrepeatarray(data=min_data, shape=x_min_vals.shape)
        max_vals = lazyrepeatarray(
            data=x_max_vals.data.__pow__(*args, **kwargs), shape=x_max_vals.shape
        )
    elif op_str == "argsort":
        min_vals = x_min_vals * 0
        max_vals = x_max_vals * 0 + np.prod(x_max_vals.shape)
    elif op_str == "cumsum":
        dummy_res = np.empty(x_min_vals.shape).cumsum(*args, **kwargs)
        num = np.ones(x_min_vals.shape).cumsum(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data * num, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data * num, shape=dummy_res.shape)
    elif op_str == "cumprod":
        dummy_res = np.empty(x_min_vals.shape).cumprod(*args, **kwargs)
        num = np.ones(x_min_vals.shape).cumsum(*args, **kwargs)
        if abs(x_max_vals.data) >= abs(x_min_vals.data):
            highest = abs(x_max_vals.data)
        else:
            highest = abs(x_min_vals.data)

        min_vals = lazyrepeatarray(
            data=-((highest**num).max()), shape=dummy_res.shape
        )
        max_vals = lazyrepeatarray(data=(highest**num).max(), shape=dummy_res.shape)
    elif op_str == "prod":
        dummy_res = np.empty(x_min_vals.shape).prod(*args, **kwargs)
        min_vals = lazyrepeatarray(
            data=x_min_vals.data ** (np.prod(x_min_vals.shape) / dummy_res.size),
            shape=dummy_res.shape,
        )
        max_vals = lazyrepeatarray(
            data=x_max_vals.data ** (np.prod(x_max_vals.shape) / dummy_res.size),
            shape=dummy_res.shape,
        )
    elif op_str == "var":
        dummy_res = np.empty(x_min_vals.shape).var(*args, **kwargs)
        min_vals = lazyrepeatarray(data=0, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(
            data=0.25 * (x_max_vals.data - x_min_vals.data) ** 2,
            shape=dummy_res.shape,
        )
    elif op_str == "take":
        dummy_res = np.empty(x_min_vals.shape).take(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "flatten":
        dummy_res = np.empty(x_min_vals.shape).flatten(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "ravel":
        dummy_res = np.empty(x_min_vals.shape).ravel(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "compress":
        dummy_res = np.empty(x_min_vals.shape).compress(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "squeeze":
        dummy_res = np.empty(x_min_vals.shape).squeeze(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "resize":
        dummy_res = np.empty(x_min_vals.shape)
        dummy_res.resize(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "transpose":
        dummy_res = np.empty(x_min_vals.shape).transpose(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "reshape":
        dummy_res = np.empty(x_min_vals.shape).reshape(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "__getitem__":
        dummy_res = np.empty(x_min_vals.shape).__getitem__(*args, **kwargs)
        min_vals = lazyrepeatarray(data=x_min_vals.data, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=x_max_vals.data, shape=dummy_res.shape)
    elif op_str == "__mod__":
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                maxv = other.max()
            else:
                maxv = other
        else:
            if hasattr(other, "min_vals") and hasattr(other, "max_vals"):
                maxv = other.max_vals.data  # type: ignore
            else:
                raise ValueError(
                    f"Not supported type for lazy repeat array computation: {type(other)}"
                )
        min_vals = lazyrepeatarray(data=0, shape=x_min_vals.shape)
        max_vals = lazyrepeatarray(data=maxv, shape=x_max_vals.shape)
    elif op_str in ["__lshift__", "__rshift__"]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other
        elif hasattr(other, "min_vals") and hasattr(other, "max_vals"):
            other_max = other.max_vals.data  # type: ignore
            other_min = other.min_vals.data  # type: ignore
        else:
            raise NotImplementedError(
                f"{op_str} not implemented in LazyRepeatArray for {type(other)}."
            )

        min_min = x_min_vals.data << other_min
        min_max = x_min_vals.data << other_max
        max_min = x_max_vals.data << other_min
        max_max = x_max_vals.data << other_max

        _min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
        _max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

        min_vals = lazyrepeatarray(data=_min_vals, shape=x_min_vals.shape)
        max_vals = lazyrepeatarray(data=_max_vals, shape=x_max_vals.shape)

    elif op_str == "__xor__":
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other
        elif hasattr(other, "min_vals") and hasattr(other, "max_vals"):
            other_max = other.max_vals.data  # type: ignore
            other_min = other.min_vals.data  # type: ignore
        else:
            raise NotImplementedError(
                f"{op_str} not implemented in LazyRepeatArray for {type(other)}."
            )

        # TODO: should modify for a tighter found for xor
        _max = int(max(x_max_vals.data, other_max))
        _min = int(min(x_min_vals.data, other_min))
        _max_vals = max(
            (2 ** (_min ^ _max).bit_length()) - 1, (2 ** (_max).bit_length()) - 1
        )
        _min_vals = min(0, _min)

        min_vals = lazyrepeatarray(data=_min_vals, shape=x_min_vals.shape)
        max_vals = lazyrepeatarray(data=_max_vals, shape=x_max_vals.shape)

    elif op_str == "ptp":
        dummy_res = np.empty(x_min_vals.shape).ptp(*args, **kwargs)
        min_vals = lazyrepeatarray(data=0, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(
            data=x_max_vals.data - x_min_vals.data, shape=dummy_res.shape
        )

    elif op_str in ["argmax", "argmin"]:
        dummy_res = np.empty(x_min_vals.shape)
        if len(args) or "axis" in kwargs:
            max_value = np.size(dummy_res, *args, **kwargs) - 1
        else:
            max_value = dummy_res.size - 1
        min_vals = lazyrepeatarray(data=0, shape=x_min_vals.shape)
        max_vals = lazyrepeatarray(data=max_value, shape=x_min_vals.shape)
    elif op_str == "__abs__":
        min_val = abs(x_min_vals.data)
        max_val = abs(x_max_vals.data)

        new_min_val = min(min_val, max_val)
        new_max_val = max(min_val, max_val)
        min_vals = lazyrepeatarray(data=new_min_val, shape=x_min_vals.shape)
        max_vals = lazyrepeatarray(data=new_max_val, shape=x_max_vals.shape)
    elif op_str in ["all", "any"]:
        dummy_res = getattr(np.empty(x_min_vals.shape), op_str)(*args, **kwargs)
        min_vals = lazyrepeatarray(data=0, shape=dummy_res.shape)
        max_vals = lazyrepeatarray(data=1, shape=dummy_res.shape)
    elif op_str == "__or__":
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other
        elif hasattr(other, "min_vals") and hasattr(other, "max_vals"):
            other_max = other.max_vals.data  # type: ignore
            other_min = other.min_vals.data  # type: ignore
        else:
            raise NotImplementedError(
                f"{op_str} not implemented in LazyRepeatArray for {type(other)}."
            )

        # TODO: should modify for a tighter found for xor
        _max = int(max(x_max_vals.data, other_max))
        _min_vals = int(min(x_min_vals.data, other_min))
        _max_vals = (2 ** (_max).bit_length()) - 1

        min_vals = lazyrepeatarray(data=_min_vals, shape=x_min_vals.shape)
        max_vals = lazyrepeatarray(data=_max_vals, shape=x_max_vals.shape)
    elif op_str == "__and__":
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                other_max, other_min = other.max(), other.min()
            else:
                other_max, other_min = other, other
        elif hasattr(other, "min_vals") and hasattr(other, "max_vals"):
            other_max = other.max_vals.data  # type: ignore
            other_min = other.min_vals.data  # type: ignore
        else:
            raise NotImplementedError(
                f"{op_str} not implemented in LazyRepeatArray for {type(other)}."
            )

        # TODO: should modify for a tighter found for and
        _max_vals = max(x_max_vals.data, other_max)
        _min = min(x_min_vals.data, other_min)
        if x_min_vals.data < 0 and other_min < 0:
            _min_vals = -(2 ** _min.bit_length())
        else:
            _min_vals = min(0, _min)

        min_vals = lazyrepeatarray(data=_min_vals, shape=x_min_vals.shape)
        max_vals = lazyrepeatarray(data=_max_vals, shape=x_max_vals.shape)

    else:
        raise ValueError(f"Invaid Operation for LazyRepeatArray: {op_str}")

    return (min_vals, max_vals)
