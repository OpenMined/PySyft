# stdlib
from typing import NewType
from typing import Union

# syft relative
from .primitive_interface import PyPrimitive

NotImplementedType = NewType("NotImplementedType", type(NotImplemented))  # type: ignore
SyPrimitiveRet = NewType("SyPrimitiveRet", Union[PyPrimitive, NotImplementedType])  # type: ignore
