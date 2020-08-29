from typing import Union

from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive
from .none import SyNone


class PrimitiveFactory:
    @staticmethod
    @syft_decorator(typechecking=True)
    def generate_primitive(value: Union[int, float, bool, None]) -> PyPrimitive:
        from .int import Int
        from .none import SyNone
        from .float import Float
        from .bool import Bool

        if type(value) is int:
            return Int(value=value)

        if type(value) is float:
            return Float(value=value)

        if type(value) is bool:
            return Bool(value=value)

        return SyNone()
