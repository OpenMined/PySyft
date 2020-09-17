# stdlib
from abc import ABC
from typing import Any
from typing import Optional
from typing import Union

# syft relative
from ...core.common import UID
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive


def isprimitive(value: Any) -> bool:
    if not issubclass(type(value), PyPrimitive) and type(value) in [
        int,
        float,
        bool,
        complex,
        list,
        str,
        None,
    ]:
        return True
    return False


class PrimitiveFactory(ABC):
    def upcast(self) -> Union[int, float, bool, complex, list, str, None]:
        raise NotImplementedError

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

        if isinstance(value, bool):
            return Bool(value=value, id=id)

        if isinstance(value, int):
            return Int(value=value, id=id)

        if isinstance(value, float):
            return Float(value=value, id=id)

        if isinstance(value, complex):
            return Complex(real=value.real, imag=value.imag, id=id)

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
