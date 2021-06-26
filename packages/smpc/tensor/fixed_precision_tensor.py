import numpy as np

from syft.core.tensor.passthrough import PassthroughTensor
from .fixed_precision_tensor_ancestor import FixedPrecisionTensorAncestor

class FixedPrecisionTensor(PassthroughTensor, FixedPrecisionTensorAncestor):
    def __init__(self, base, precision, value):
        self._base = base
        self._precision = precision
        self._scale = base ** precision
        encoded_value = (self._scale * value).astype(np.int64)
        super().__init__(encoded_value)


    def decode(self):
        correction = (self.child < 0).astype(np.int64)
        dividend = self.child // self._scale - correction
        remainder = self.child % self._scale
        remainder += (remainder == 0).astype(np.int64) * self._scale * correction
        value = dividend.astype(np.float32) + remainder.astype(np.float32) / self._scale
        return value
