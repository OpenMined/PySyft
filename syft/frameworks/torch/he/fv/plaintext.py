class PlainText:
    """A wrapper class for representing plaintext.

    Typical format of plaintext data would be [x0, x1, x2...] where xi represents
    coefficients of the polynomial.

    Attributes:
        data: A 1-dim list representing plaintext coefficient values.
    """

    def __init__(self, data):
        self.data = data
