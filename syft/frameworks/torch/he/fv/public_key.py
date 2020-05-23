class PublicKey:
    """A wrapper class for representing public_key.

    Attributes:
        data: A list of values of the public key.

    Typical format:
    [c0, c1] : where c0 represents `-(a*s + e)(mod q)` and c1 represents `a`
    """

    def __init__(self, data):
        self.data = data
