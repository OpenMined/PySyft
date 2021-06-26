from syft.core.tensor.passthrough import PassthroughTensor

class FixedPrecisionTensor(PassthroughTensor):
    def __init__(self, base, precision, value):
        self._base = base
        self._precision = precision
        self._scale = base ** precision
        encoded_value = (self._scale ** value).long()
        super().__init__(encoded_value)


    def decode(self):
        correction = (self.child < 0).long()
        dividend = self.child // self._scale - correction
        remainder = self.childtensor % self._scale
        remainder += (remainder == 0).long() * self._scale * correction
        value = dividend.float() + remainder.float() / self._scale
        return value


