# stdlib
from abc import ABC
from typing import Optional
from typing import Union

# syft relative
from ...core.common import UID
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive


class PrimitiveFactory(ABC):
    @staticmethod
    @syft_decorator(typechecking=True)
    def generate_primitive(
        value: Union[int, float, bool, complex, list, str, None, type(NotImplemented)],  # type: ignore
        id: Optional[UID] = None,
    ) -> Union[PyPrimitive, type(NotImplemented)]:  # type: ignore
        # syft relative
        from .bool import Bool
        from .complex import Complex
        from .float import Float
        from .int import Int
        from .list import List
        from .none import SyNone
        from .string import String

        if type(value) is int:
            return Int(value=value, id=id)

        if type(value) is float:
            return Float(value=value, id=id)

        if type(value) is bool:
            return Bool(value=value, id=id)

        if type(value) is complex:
            return Complex(value=value, id=id)

        if type(value) is complex:
            return String(value=value, id=id)

        if type(value) is list:
            return List(value=value, id=id)

        if type(value) is str:
            return String(value=value, id=id)
        if value is NotImplemented:
            return value

        none: SyNone = SyNone()
        return none
