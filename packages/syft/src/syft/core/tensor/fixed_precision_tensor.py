# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
import numpy as np

# relative
from ..common.serde.serializable import serializable
from .passthrough import PassthroughTensor  # type: ignore
from .passthrough import is_acceptable_simple_type  # type: ignore


@serializable(recursive_serde=True)
class FixedPrecisionTensor(PassthroughTensor):

    __attr_allowlist__ = ("child", "_base", "_precision", "_scale")

    def __init__(
        self, value: Optional[Any] = None, base: int = 10, precision: int = 3
    ) -> None:

        self._base = base
        self._precision = precision
        self._scale = base**precision
        if value is not None:
            # TODO :Should modify to be compatiable with torch.
            value = np.array(value, dtype=np.int32)
            fpt_value = self._scale * value
            encoded_value = fpt_value.astype(np.int32)
            super().__init__(encoded_value)
        else:
            super().__init__(None)

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

        correction = (value < 0).astype(np.int32)
        dividend = value // self._scale - correction
        remainder = value % self._scale
        remainder += (remainder == 0).astype(np.int32) * self._scale * correction
        value = dividend.astype(np.float32) + remainder.astype(np.float32) / self._scale
        return value

    def sanity_check(
        self, other: Union[FixedPrecisionTensor, int, float, np.ndarray]
    ) -> FixedPrecisionTensor:
        if isinstance(other, FixedPrecisionTensor):
            if self.base != other.base or self.precision != other.precision:
                raise ValueError(
                    f"Base:{self.base,other.base} and Precision:"
                    + f"{self.precision, other.precision} should be same for"
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
        other = self.sanity_check(other)
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child + other.child
        return res

    def __sub__(self, other: Any) -> FixedPrecisionTensor:
        other = self.sanity_check(other)
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child - other.child
        return res

    def __mul__(self, other: Any) -> FixedPrecisionTensor:
        other = self.sanity_check(other)
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child * other.child
        return res
