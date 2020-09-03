from typing import Union

from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive


class PrimitiveFactory:
    @staticmethod
    @syft_decorator(typechecking=True)
    def generate_primitive(
        value: Union[int, float, bool, complex, None]
    ) -> PyPrimitive:
        from .int import Int
        from .none import SyNone
        from .float import Float
        from .bool import Bool
        from .complex import Complex

        if isinstance(value, int):
            return Int(value=value)

        if isinstance(value, float):
            return Float(value=value)

        if isinstance(value, bool):
            return Bool(value=value)

        if isinstance(value, complex):
            return Complex(value=value)

        return SyNone()
