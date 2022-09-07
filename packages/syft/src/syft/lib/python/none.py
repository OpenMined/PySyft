# stdlib
from typing import Any
from typing import Optional

# relative
from ...core.common import UID
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet

NoneType = type(None)


class _SyNone(PyPrimitive):
    __name__ = "_SyNone"

    def __init__(self, id: Optional[UID] = None, temporary_box: bool = False):
        self._id: UID = id if id else UID()
        self.temporary_box = temporary_box

    def upcast(self) -> NoneType:
        return None

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        if isinstance(other, _SyNone):
            return PrimitiveFactory.generate_primitive(value=True)

        if other is None:
            return PrimitiveFactory.generate_primitive(value=True)

        res = None.__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __hash__(self) -> SyPrimitiveRet:
        res = None.__hash__()
        return PrimitiveFactory.generate_primitive(value=res)


SyNone = _SyNone()
