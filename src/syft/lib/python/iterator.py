# syft relative
from ...core.common.uid import UID
from .primitive_interface import PyPrimitive


class Iterator(PyPrimitive):
    def __init__(self, _ref):
        super().__init__()
        self._obj_ref = _ref
        self._index = 0
        self._id = UID()
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration

        if self._index >= len(self._obj_ref):
            self._exhausted = True
            raise StopIteration

        obj = self._obj_ref[self._index]
        self._index += 1
        return obj
