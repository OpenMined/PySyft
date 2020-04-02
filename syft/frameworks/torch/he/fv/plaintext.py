from syft.frameworks.torch.he.fv.util.operations import get_significant_count


class PlainText:
    """Class to store a plaintext element. The data for the plaintext is a polynomial
    with coefficients modulo the plaintext modulus. The degree of the plaintext
    polynomial must be one less than the degree of the polynomial modulus."""

    def __init__(self, coeff_count, data):
        self.coeff_count = coeff_count
        self.data = data

    def resize(self, coeff_count):
        """Resizes the plaintext to have a given coefficient count.

        Args:
            coeff_count: The number of coefficients in the plaintext polynomial
        """
        self.data = [0] * coeff_count

    def set_zero(self):
        """Sets the plaintext polynomial to zero."""
        self.data = [0] * self.coeff_count

    def significant_coeff_count(self):
        """Returns the significant coefficient count of the current plaintext polynomial."""
        return get_significant_count(self.data, self.coeff_count)

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data
