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
        value: Union[int, float, bool, complex, None], id: Optional[UID] = None
    ) -> PyPrimitive:
        # syft relative
        from .bool import Bool
        from .complex import Complex
        from .float import Float
        from .int import Int
        from .none import SyNone

        if type(value) is int:
            return Int(value=value, id=id)

        if type(value) is float:
            return Float(value=value, id=id)

        if type(value) is bool:
            return Bool(value=value, id=id)

        if type(value) is complex:
            return Complex(value=value, id=id)

        none: SyNone = SyNone()
        return none
