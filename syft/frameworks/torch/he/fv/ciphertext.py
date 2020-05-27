class CipherText:
    """A wrapper class for representing ciphertext.

    Typical format of ciphertext data would be [c0, c1, c2...] where ci represents
    list of polynomials.

    Attributes:
        data: A 3-dim list representing ciphertext values.
    """

    def __init__(self, data):
        self.data = data
