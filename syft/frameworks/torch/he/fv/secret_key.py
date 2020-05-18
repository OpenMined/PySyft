class SecretKey:
    """A wrapper class for representing secret_key.

    Attributes:
        data: A list of values of the secret key.

    Typical format:
    [x1. x2, x3...] : where xi is integer denoting -1, 0, 1 in respective modulus.
    1 is represented as 1
    0 is represented as 0
    -1 is represented as (modulus-1)
    """

    def __init__(self, data):
        self.data = data
