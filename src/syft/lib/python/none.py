# syft relative
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive


class SyNone(PyPrimitive):
    pass


PrimitiveFactory.register_primitive(python_primitive=None, syft_primitive=SyNone)
PrimitiveFactory.register_default(syft_primitive=SyNone)
