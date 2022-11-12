# stdlib
from typing import Any

# relative
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive


def downcast(value: Any, recurse: bool = True) -> Any:
    if isprimitive(value=value):
        # Wrap in a SyPrimitive
        return PrimitiveFactory.generate_primitive(value=value, recurse=recurse)
    else:
        return value


def upcast(value: Any) -> Any:
    upcast_method = getattr(value, "upcast", None)
    if upcast_method is not None:
        return upcast_method()
    return value
