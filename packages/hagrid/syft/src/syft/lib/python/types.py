# stdlib
from collections import UserDict
from collections import UserList
from collections import UserString
from typing import NewType
from typing import Union

# relative
from .primitive_interface import PyPrimitive

NotImplementedType = NewType("NotImplementedType", type(NotImplemented))  # type: ignore
ImplementedInterfaces = Union[UserDict, UserList, UserString]
ImplementedSingletons = Union[type(None)]  # type: ignore
PrimitiveRetType = Union[
    PyPrimitive,
    NotImplementedType,
    ImplementedInterfaces,
    ImplementedSingletons,  # type: ignore
]
SyPrimitiveRet = NewType("SyPrimitiveRet", PrimitiveRetType)  # type: ignore
