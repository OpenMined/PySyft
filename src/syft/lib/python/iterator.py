# stdlib
from typing import Any
from typing import Optional

# syft relative
from ...core.common.uid import UID
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive


class Iterator(PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, _ref: Any, max_len: Optional[int] = None):
        super().__init__()
        self._obj_ref = _ref
        self._index = 0
        self._id = UID()
        self.max_len = max_len if max_len is not None else len(self._obj_ref)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iter__(self) -> "Iterator":
        return self

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __next__(self) -> Any:
        if self._index >= self.max_len:
            raise StopIteration

        if hasattr(self._obj_ref, "__next__"):
            obj = next(self._obj_ref)
        elif hasattr(self._obj_ref, "__getitem__"):
            obj = self._obj_ref[self._index]
        else:
            raise ValueError("Can't iterate through given object.")

        self._index += 1
        return obj
