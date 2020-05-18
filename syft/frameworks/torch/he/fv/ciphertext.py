class CipherText:
    """A wrapper class for representing ciphertext.

    Attributes:
        data: A list of lists of integers representing ciphertext.

    Typical format:
    [c0, c1, c2...] where ci represents polynomials(list of integers).
    """

    def __init__(self, data):
        self.data = data
