class CipherText:
    """A wrapper class for representing ciphertext.

    Typical format of ciphertext data would be [c0, c1, c2...] where ci represents
    list of polynomials.

    Attributes:
        data: A 3-dim list representing ciphertext values.
        param_id: parameter id used in encryption process.
    """

    def __init__(self, data, param_id):
        self.data = data
        self.param_id = param_id
