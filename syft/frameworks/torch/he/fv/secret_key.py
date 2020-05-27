class SecretKey:
    """A wrapper class for representing secret key.

    Typical format of secret key data would be [p1. p2, p3...] where pi represents
    polynomials for each coefficient modulus.

    Elements of each polynomails is taken from {-1, 0, 1} represented in their respective
    modulus.

    Attributes:
        data: A 2-dim list representing secret key values.
    """

    def __init__(self, data):
        self.data = data
