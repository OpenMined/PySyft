class PublicKey:
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def param(self, value):
        self._data = value
