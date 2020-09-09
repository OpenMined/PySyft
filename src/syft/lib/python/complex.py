# stdlib
from typing import Any
from typing import Optional

# syft relative
from ...core.common import UID
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive


# TODO - actually make all of this work
class Complex(complex, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(self, value: Any = None, id: Optional[UID] = None) -> "Complex":
        if value is None:
            value = 0.0
        return complex.__new__(self, value)  # type: ignore

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Any = None, id: Optional[UID] = None):
        if value is None:
            value = 0.0

        complex.__init__(value)

        if id is None:
            self._id = UID()
        else:
            self._id = id
