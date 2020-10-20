from .primitive_interface import PyPrimitive
from ...core.common.uid import UID

class Iterator(PyPrimitive):
    def __init__(self, _ref):
        super().__init__()
        self._ref = _ref
        self._index = 0
        self._id = UID()

    def __next__(self):

