# stdlib
from typing import Union

# syft relative
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive


class PrimitiveFactory:
    @staticmethod
    @syft_decorator(typechecking=True)
    def generate_primitive(
        value: Union[int, float, bool, complex, None]
    ) -> PyPrimitive:
        # syft relative
        from .bool import Bool
        from .complex import Complex
        from .float import Float
        from .int import Int
        from .none import SyNone

        if type(value) is int:
            return Int(value=value)

        if type(value) is float:
            return Float(value=value)

        if type(value) is bool:
            return Bool(value=value)

        if type(value) is complex:
            return Complex(value=value)

        return SyNone()
