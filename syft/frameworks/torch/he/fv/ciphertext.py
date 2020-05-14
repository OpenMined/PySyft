class CipherText:
    """A wrapper class for representing ciphertext.

    Attributes:
        data: A list of values of ciphertext.
    """

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data
