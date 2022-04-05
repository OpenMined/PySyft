# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
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
        self, value: Optional[Any] = None, base: int = 2, precision: int = 16
    ) -> None:

        self._base = base
        self._precision = precision
        self._scale = base**precision
        if value is not None:
            # TODO :Should modify to be compatiable with torch.

            super().__init__(self.encode(value))
        else:
            super().__init__(None)

    def encode(self, value) -> Any:
        value = np.array(value, DEFAULT_INT_NUMPY_TYPE)
        fpt_value = self._scale * value
        encoded_value = fpt_value.astype(DEFAULT_INT_NUMPY_TYPE)
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

    def decode(self) -> Any:
        # relative
        from .smpc.share_tensor import ShareTensor

        value = self.child.child if isinstance(self.child, ShareTensor) else self.child

        correction = (value < 0).astype(DEFAULT_INT_NUMPY_TYPE)
        dividend = value // self._scale - correction
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

    def __truediv__(
        self, other: Union[int, np.integer, FixedPrecisionTensor]
    ) -> FixedPrecisionTensor:
        if isinstance(other, FixedPrecisionTensor):
            raise ValueError("We do not support Private Division yet.")

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
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

    # TODO: Remove after moving private compare to sharetensor level
    def __lt__(self, other) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child < other.child
        return res
