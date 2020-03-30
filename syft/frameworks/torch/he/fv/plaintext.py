from syft.frameworks.torch.he.fv.util.operations import get_significant_count


class PlainText:
    """Class to store a plaintext element. The data for the plaintext is a polynomial
    with coefficients modulo the plaintext modulus. The degree of the plaintext
    polynomial must be one less than the degree of the polynomial modulus."""

    def __init__(self, coeff_count, data):
        self.coeff_count = coeff_count
        self.data = data

    def significant_coeff_count(self):
        return get_significant_count(self.data, self.coeff_count)

    @property
    def data(self):
        return self.data
