class PublicKey:
    """A wrapper class for representing public key.

    Typical format of public key data would be [c0, c1, ...] where ci represents
    a list of polynomials.

    Attributes:
        data: A 3-dim list representing public key values.
    """

    def __init__(self, data):
        self.data = data
