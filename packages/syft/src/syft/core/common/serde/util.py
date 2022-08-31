# stdlib
import sys
from typing import Any

# relative
from ....lib.util import full_name_with_qualname


def serialize_type(serialized_type: Any) -> str:
    fqn = full_name_with_qualname(klass=serialized_type)
    module_parts = fqn.split(".")
    _ = module_parts.pop()  # remove incorrect .type ending
    module_parts.append(serialized_type.__name__)
    return ".".join(module_parts)


def deserialize_type(deserialized_type: str) -> type:
    module_parts = deserialized_type.split(".")
    klass = module_parts.pop()
    exception_type = getattr(sys.modules[".".join(module_parts)], klass)
    return exception_type
