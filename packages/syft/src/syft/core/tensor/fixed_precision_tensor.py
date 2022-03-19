# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional

# third party
import numpy as np

# relative
from ..common.serde.serializable import serializable
from .passthrough import PassthroughTensor  # type: ignore


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
            fpt_value = self._scale * value
            encoded_value = fpt_value.astype(np.int32)
            super().__init__(encoded_value)

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
        correction = (self.child < 0).astype(np.int32)
        dividend = self.child // self._scale - correction
        remainder = self.child % self._scale
        remainder += (remainder == 0).astype(np.int32) * self._scale * correction
        value = dividend.astype(np.float32) + remainder.astype(np.float32) / self._scale
        return value

    def __add__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child + other.child
        return res

    def __sub__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child - other.child
        return res
