# stdlib
from abc import ABC
from collections import UserDict
from collections import UserList
from collections import UserString
from typing import Any
from typing import Optional
from typing import Union

# syft relative
from .. import python
from ...core.common import UID
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive

NoneType = type(None)

primitives = [
    bool,
    dict,
    complex,
    float,
    int,
    list,
    None,
    NoneType,
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
    NoneType,
    str,
    UserDict,
    UserList,
    UserString,
]


@syft_decorator(typechecking=True)
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
        recurse: bool = False,
    ) -> Union[PyPrimitive, type(NotImplemented)]:  # type: ignore
        # syft relative

        if isinstance(value, PyPrimitive):
            return value

        if isinstance(value, bool):
            return python.Bool(value=value, id=id)

        if isinstance(value, int):
            return python.Int(value=value, id=id)

        if isinstance(value, float):
            return python.Float(value=value, id=id)

        if isinstance(value, complex):
            return python.Complex(real=value.real, imag=value.imag, id=id)

        if type(value) in [list, UserList]:
            if not recurse:
                return python.List(value=value, id=id)
            else:
                # allow recursive primitive downcasting
                new_list = []
                if value is not None:
                    for val in value:
                        if isprimitive(value=val):
                            new_list.append(
                                PrimitiveFactory.generate_primitive(
                                    value=val, recurse=recurse
                                )
                            )
                        else:
                            new_list.append(val)
                return python.List(value=new_list, id=id)

        if type(value) in [dict, UserDict]:
            if not recurse:
                new_dict = python.Dict(value)
            else:
                # allow recursive primitive downcasting
                new_dict = python.Dict()
                if value is not None:
                    items = getattr(value, "items", None)
                    if items is not None:
                        for k, val in items():
                            if isprimitive(value=val):
                                new_dict[k] = PrimitiveFactory.generate_primitive(
                                    value=val, recurse=recurse
                                )
                            else:
                                new_dict[k] = val
            # if we pass id in as a kwargs it ends up in the actual dict
            if id is not None:
                new_dict._id = id
            return new_dict

        if type(value) in [str, UserString]:
            return python.String(value=value, id=id)

        if value is NotImplemented:
            return value

        none: python.SyNone = python.SyNone()
        return none
