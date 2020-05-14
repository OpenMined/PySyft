class SecretKey:
    """A wrapper class for representing secret_key.

    Attributes:
        data: A list of values of secret key.
    """

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data
