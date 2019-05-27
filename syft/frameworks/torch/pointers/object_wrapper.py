class ObjectWrapper:
    def __init__(self, obj, id: int):
        self._obj = obj
        self.id = id

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __str__(self):
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " id:" + str(self.id)
        out += " obj:" + str(self._obj)
        out += ">"
        return out

    def __repr__(self):
        return str(self)

    @property
    def obj(self):
        return self._obj
