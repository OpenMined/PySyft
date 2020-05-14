import torch as th
from syft.frameworks.torch.he.fv.util.operations import get_significant_count


class PlainText:
    """A wrapper class for representing plaintext elements.

    Attributes:
        data: A list of values of plaintext polynomial.
    """

    def __init__(self, data):
        self._data = data
        self._coeff_count = len(data)

    def significant_coeff_count(self):
        """Returns the significant coefficient count of the polynomial (removes leading zeroes)"""
        return get_significant_count(self._data)

    @property
    def data(self):
        return self._data

    @property
    def coeff_count(self):
        return self._coeff_count
