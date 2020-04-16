class SecretKey:
    def __init__(self, data):
        self._data = data
        self._size = len(self.data)

    @property
    def data(self):
        return self._data

    @data.setter
    def param(self, value):
        self._data = value

    @property
    def size(self):
        return self._size
