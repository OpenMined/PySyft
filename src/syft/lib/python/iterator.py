from .primitive_interface import PyPrimitive
from ...core.common.uid import UID

class Iterator(PyPrimitive):
    def __init__(self, _ref):
        super().__init__()
        self._obj_ref = _ref
        self._index = 0
        self._id = UID()

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._obj_ref):
            raise StopIteration

        obj = self._obj_ref[self._index]
        self._index += 1
        return obj
