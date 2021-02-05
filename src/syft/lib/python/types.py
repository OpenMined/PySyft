from typing import Union
from typing import NewType
from .primitive_interface import PyPrimitive
from collections import UserList, UserDict, UserString

NotImplementedType = NewType("NotImplementedType", type(NotImplemented))  # type: ignore
ImplementedInterfaces = Union[UserDict, UserList, UserString]
ImplementedSingletons = Union[type(None)]  # type: ignore
PrimitiveRetType = Union[PyPrimitive, NotImplementedType, ImplementedInterfaces]
SyPrimitiveRet = NewType("SyPrimitiveRet", PrimitiveRetType)  # type: ignore
