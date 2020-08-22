import torch

from .lowercase_tensor import LowercaseTensorConstructor
from .uppercase_tensor import UppercaseTensorConstructor
from .parameter import ParameterConstructor

__all__ = [
    "LowercaseTensorConstructor",
    "UppercaseTensorConstructor",
    "ParameterConstructor",
]

from syft.ast.globals import Globals

from .allowlist import allowlist


def create_torch_ast() -> Globals:
    ast = Globals()

    for method, return_type_name in allowlist.items():
        ast.add_path(
            path=method, framework_reference=torch, return_type_name=return_type_name
        )

    for klass in ast.classes:

        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()
    return ast
