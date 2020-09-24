# stdlib
from abc import ABC
from collections import UserDict
from collections import UserList
from collections import UserString
from typing import Any
from typing import Optional
from typing import Union

# syft relative
from ...core.common import UID
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive

primitives = [
    bool,
    dict,
    complex,
    float,
    int,
    list,
    None,
    str,
    UserDict,
    UserList,
    UserString,
]

PrimitiveType = Union[
    bool,
    dict,
    complex,
    float,
    int,
    list,
    None,
    str,
    UserDict,
    UserList,
    UserString,
]


def isprimitive(value: Any) -> bool:
    if not issubclass(type(value), PyPrimitive) and type(value) in primitives:
        return True
    return False


class PrimitiveFactory(ABC):
    def upcast(self) -> Union[int, float, bool, complex, list, str, None]:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def generate_primitive(
        value: Union[PrimitiveType, type(NotImplemented), PyPrimitive],  # type: ignore
        id: Optional[UID] = None,
    ) -> Union[PyPrimitive, type(NotImplemented)]:  # type: ignore
        # syft relative
        from .bool import Bool
        from .complex import Complex
        from .dict import Dict
        from .float import Float
        from .int import Int
        from .list import List
        from .none import SyNone
        from .string import String

        if isinstance(value, PyPrimitive):
            return value

        if isinstance(value, bool):
            return Bool(value=value, id=id)

        if isinstance(value, int):
            return Int(value=value, id=id)

        if isinstance(value, float):
            return Float(value=value, id=id)

        if isinstance(value, complex):
            return Complex(real=value.real, imag=value.imag, id=id)

        if type(value) in [list, UserList]:
            return List(value=value, id=id)

        if type(value) in [dict, UserDict]:
            new_dict = Dict(value)
            # if we pass id in as a kwargs it ends up in the actual dict
            if id is not None:
                new_dict._id = id
            return new_dict

        if type(value) in [str, UserString]:
            return String(value=value, id=id)

        if value is NotImplemented:
            return value

        none: SyNone = SyNone()
        return none
