class PublicKey:
    """A wrapper class for representing public_key.

    Attributes:
        data: A list of values of public key.
    """

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data
