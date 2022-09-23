# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np

# relative
from ..common.serde.serializable import serializable
from .config import DEFAULT_FLOAT_NUMPY_TYPE
from .config import DEFAULT_INT_NUMPY_TYPE
from .passthrough import PassthroughTensor  # type: ignore
from .passthrough import is_acceptable_simple_type  # type: ignore
from .smpc import context


@serializable(recursive_serde=True)
class FixedPrecisionTensor(PassthroughTensor):

    __attr_allowlist__ = ("child", "_base", "_precision", "_scale")

    def __init__(
        self,
        value: Union[int, float, np.ndarray] = None,
        base: int = 2,
        precision: int = 16,
    ) -> None:
        self._base = base
        self._precision = precision
        self._scale = base**precision
        if value is not None:
            # TODO :Should modify to be compatiable with torch.

            super().__init__(self.encode(value))
        else:
            super().__init__(None)

    def encode(self, value: Union[int, float, np.ndarray]) -> np.ndarray:
        encoded_value = np.array(self._scale * value, DEFAULT_INT_NUMPY_TYPE)
        return encoded_value

    @property
    def precision(self) -> int:
        """Get the precision for the FixedPrecisionTensor.

        Returns:
            int: precision.
        """
        return self._precision

    @property
    def base(self) -> int:
        """Get the base for the FixedPrecisionTensor.

        Returns:
            int: base
        """
        return self._base

    @property
    def scale(self) -> int:
        """Get the scale for the FixedPrecisionTensor.

        Returns:
            int: the scale.
        """
        return self._scale

    @property
    def dtype(self) -> np.dtype:
        return getattr(self.child, "dtype", None)

    @property
    def shape(self) -> Tuple[Any, ...]:
        if self.child is None:
            return ()
        else:
            return self.child.shape

    def decode(self) -> Any:
        # relative
        from .smpc.share_tensor import ShareTensor

        value = self.child.child if isinstance(self.child, ShareTensor) else self.child

        correction = (value < 0).astype(DEFAULT_INT_NUMPY_TYPE)

        dividend = np.trunc(value / self._scale - correction)
        remainder = value % self._scale
        remainder += (
            (remainder == 0).astype(DEFAULT_INT_NUMPY_TYPE) * self._scale * correction
        )
        value = (
            dividend.astype(DEFAULT_FLOAT_NUMPY_TYPE)
            + remainder.astype(DEFAULT_FLOAT_NUMPY_TYPE) / self._scale
        )
        return value

    def sanity_check(
        self, other: Union[FixedPrecisionTensor, int, float, np.ndarray]
    ) -> FixedPrecisionTensor:
        if isinstance(other, FixedPrecisionTensor):
            if self.base != other.base or self.precision != other.precision:
                raise ValueError(
                    f"Base:{self.base,other.base} and Precision: "
                    + f"{self.precision, other.precision} should be same for "
                    + "computation on FixedPrecisionTensor"
                )
        elif is_acceptable_simple_type(other):
            other = FixedPrecisionTensor(
                value=other, base=self.base, precision=self.precision
            )
        else:
            raise ValueError(f"Invalid type for FixedPrecisionTensor: {type(other)}")

        return other

    def __add__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            res.child = self.child + other
        else:
            other = self.sanity_check(other)
            res.child = self.child + other.child
        return res

    def __sub__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            res.child = self.child - other
        else:
            other = self.sanity_check(other)
            res.child = self.child - other.child
        return res

    def __mul__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            res.child = self.child * other
        else:
            other = self.sanity_check(other)
            context.FPT_CONTEXT["seed_id_locations"] = context.SMPC_CONTEXT.get(
                "seed_id_locations", None
            )
            res.child = self.child * other.child
            res = res / self.scale

        return res

    def __matmul__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            raise ValueError("Should not get  a boolan array to matmul")
        else:
            other = self.sanity_check(other)
            context.FPT_CONTEXT["seed_id_locations"] = context.SMPC_CONTEXT.get(
                "seed_id_locations", None
            )
            res.child = self.child @ other.child
            res = res / self.scale

        return res

    def __rmatmul__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            raise ValueError("Should not get  a boolan array to matmul")
        else:
            other = self.sanity_check(other)
            context.FPT_CONTEXT["seed_id_locations"] = context.SMPC_CONTEXT.get(
                "seed_id_locations", None
            )
            res.child = self.child.__rmatmul__(other.child)
            res = res / self.scale

        return res

    def __truediv__(
        self, other: Union[int, np.integer, FixedPrecisionTensor]
    ) -> FixedPrecisionTensor:
        if isinstance(other, FixedPrecisionTensor):
            raise ValueError("We do not support Private Division yet.")

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(self.child, np.ndarray) or np.isscalar(self.child):
            res.child = np.trunc(self.child / other).astype(DEFAULT_INT_NUMPY_TYPE)
        else:
            res.child = self.child / other
        return res

    def transpose(
        self, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child.transpose(*args, **kwargs)
        return res

    @property
    def T(self) -> FixedPrecisionTensor:
        return self.transpose()

    # For comparison , we could skip conversion to FPT , as operation holds without encoding.
    def __lt__(self, other: Any) -> FixedPrecisionTensor:
        other = self.sanity_check(other)
        value = (self.child < other.child) * 1

        res = FixedPrecisionTensor(
            value=value, base=self._base, precision=self._precision
        )
        return res

    def __gt__(self, other: Any) -> FixedPrecisionTensor:
        other = self.sanity_check(other)
        value = (self.child > other.child) * 1
        res = FixedPrecisionTensor(
            value=value, base=self._base, precision=self._precision
        )
        return res

    def concatenate(
        self, other: FixedPrecisionTensor, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> FixedPrecisionTensor:
        if not isinstance(other, FixedPrecisionTensor):
            raise NotImplementedError

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child.concatenate(other.child, *args, **kwargs)

        return res

    def all(self) -> bool:
        return self.child.all()

    def copy(self, order: Optional[str] = "K") -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child.copy(order=order)
        return res

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(self.child, np.ndarray):
            res.child = np.array(self.child.sum(axis=axis))
        else:
            res.child = self.child.sum(axis=axis)
        return res

    def __getitem__(
        self, item: Union[str, int, slice, PassthroughTensor]
    ) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self.base, precision=self.precision)
        if isinstance(item, PassthroughTensor):
            res.child = self.child.getitem(item.child)
        else:
            res.child = self.child.getitem(item)

        return res

    def one_hot(self) -> FixedPrecisionTensor:
        if not isinstance(self.child, np.ndarray):
            raise ValueError("One hot work only for numpy array child type")

        value = self.decode().astype(DEFAULT_INT_NUMPY_TYPE)
        one_hot_Y = np.zeros((value.size, value.max() + 1))
        one_hot_Y[np.arange(value.size), value] = 1
        one_hot_Y = one_hot_Y.T

        res = FixedPrecisionTensor(
            value=one_hot_Y, base=self.base, precision=self.precision
        )
        return res
