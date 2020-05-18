class PublicKey:
    """A wrapper class for representing public_key.

    Attributes:
        data: A list of values of the public key.

    Typical format:
    [ct0, ct1] : where ct0 represents `-(a*s + e)(mod q)` and ct1 represents `a`
    """

    def __init__(self, data):
        self.data = data
