import torch as th
from syft.frameworks.torch.he.fv.util.operations import get_significant_count


class PlainText:
    """Class to store a plaintext element. The data for the plaintext is a polynomial
    with coefficients modulo the plaintext modulus. The degree of the plaintext
    polynomial must be one less than the degree of the polynomial modulus."""

    def __init__(self, coeff_count, data):
        self._coeff_count = coeff_count
        self._data = data

    def resize(self, coeff_count):
        self.data = th.zeros(coeff_count, dtype=th.int64)

    def set_zero(self):
        """Sets the plaintext polynomial to zero."""
        self._data = [0] * self._coeff_count
        self.data = th.zeros(self._coeff_count, dtype=th.int64)

    def significant_coeff_count(self):
        """Returns the significant coefficient count of the current plaintext polynomial."""
        return get_significant_count(self._data, self._coeff_count)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def coeff_count(self):
        return self._coeff_count

    @coeff_count.setter
    def coeff_count(self, coeff_count):
        self._coeff_count = coeff_count
