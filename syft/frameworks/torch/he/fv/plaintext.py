from syft.frameworks.torch.he.fv.util.operations import get_significant_count


class PlainText:
    """A wrapper class for representing plaintext elements.

    Attributes:
        data: A list of coefficient values of plaintext polynomial.

    Typical format:
    [c0, c1, c2...] where ci represents the coefficient of the polynomial.
    """

    def __init__(self, data):
        self.data = data
        self.coeff_count = len(data)

    def significant_coeff_count(self):
        """Returns the significant coefficient count of the polynomial (removes leading zeroes)"""
        return get_significant_count(self.data)
