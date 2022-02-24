# stdlib
from abc import ABC
from collections import OrderedDict
from collections import UserDict
from collections import UserList
from collections import UserString
from typing import Any
from typing import Optional
from typing import Union

# relative
from .. import python
from ...core.common import UID
from ...logger import traceback_and_raise
from .primitive_interface import PyPrimitive

NoneType = type(None)

primitives = [
    bool,
    dict,
    complex,
    float,
    int,
    list,
    tuple,
    set,
    slice,
    range,
    None,
    NoneType,
    str,
    UserDict,
    UserList,
    UserString,
    OrderedDict,
    bytes,
]

PrimitiveType = Union[
    bool,
    dict,
    complex,
    float,
    int,
    tuple,
    list,
    set,
    slice,
    range,
    None,
    NoneType,
    str,
    UserDict,
    UserList,
    UserString,
    OrderedDict,
    bytes,
]


def isprimitive(value: Any) -> bool:
    if not issubclass(type(value), PyPrimitive) and type(value) in primitives:
        return True
    return False


class PrimitiveFactory(ABC):
    def upcast(self) -> Union[int, float, bool, complex, list, str, None]:
        traceback_and_raise(NotImplementedError)

    @staticmethod
    def generate_primitive(
        value: Union[PrimitiveType, type(NotImplemented), PyPrimitive],  # type: ignore
        id: Optional[UID] = None,
        recurse: bool = False,
        temporary_box: bool = False,
    ) -> Any:
        if isinstance(value, PyPrimitive):
            return value

        if isinstance(value, bool):
            return python.Bool(value=value, id=id)

        if isinstance(value, int):
            return python.Int(value=value, id=id)

        if isinstance(value, bytes):
            return python.Bytes(value=value)

        if isinstance(value, float):
            return python.Float(value=value, id=id)

        if isinstance(value, complex):
            return python.Complex(real=value.real, imag=value.imag, id=id)

        if isinstance(value, tuple):
            return python.Tuple(value)

        if isinstance(value, set):
            return python.Set(value)

        if isinstance(value, slice):
            return python.Slice(
                start=value.start, stop=value.stop, step=value.step, id=id
            )

        if isinstance(value, range):
            return python.Range(
                start=value.start, stop=value.stop, step=value.step, id=id
            )

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
                                    value=val,
                                    recurse=recurse,
                                    temporary_box=temporary_box,
                                )
                            )
                        else:
                            new_list.append(val)
                return python.List(value=new_list, id=id)

        if type(value) in [dict, UserDict, OrderedDict]:
            constructor = (
                python.collections.OrderedDict
                if type(value) is OrderedDict
                else python.Dict
            )

            if not recurse:
                new_dict = constructor(value)
            else:
                # allow recursive primitive downcasting
                new_dict = constructor()
                if value is not None:
                    items = getattr(value, "items", None)
                    if items is not None:
                        for k, val in items():
                            if isprimitive(value=val):
                                new_dict[k] = PrimitiveFactory.generate_primitive(
                                    value=val,
                                    recurse=recurse,
                                    temporary_box=temporary_box,
                                )
                            else:
                                new_dict[k] = val
            # if we pass id in as a kwargs it ends up in the actual dict
            if id is not None:
                new_dict._id = id
            return new_dict

        if type(value) in [str, UserString]:
            return python.String(value=value, id=id, temporary_box=temporary_box)

        if value is NotImplemented:
            return value

        return python.SyNone
