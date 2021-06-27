# third party
import numpy as np

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor


class FixedPrecisionTensor(PassthroughTensor):
    def __init__(self, value=None, base=10, precision=3):
        self._base = base
        self._precision = precision
        self._scale = base ** precision
        if value is not None:
            encoded_value = (self._scale * value).astype(np.int64)
            super().__init__(encoded_value)

    def decode(self):
        correction = (self.child < 0).astype(np.int64)
        dividend = self.child // self._scale - correction
        remainder = self.child % self._scale
        remainder += (remainder == 0).astype(np.int64) * self._scale * correction
        value = dividend.astype(np.float32) + remainder.astype(np.float32) / self._scale
        return value

    def __add__(self, other):
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child + other.child
        return res

    def __sub__(self, other):
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child - other.child
        return res
