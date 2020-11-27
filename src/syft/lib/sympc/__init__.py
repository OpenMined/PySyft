# stdlib
from typing import Dict
from typing import Union

# third party
import sympc

# syft relative
from . import session  # noqa: 401
from . import share  # noqa: 401
from ...ast.globals import Globals
from .allowlist import allowlist


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def create_sympc_ast() -> Globals:

    ast = Globals()

    for method, return_type_name_or_dict in allowlist.items():
        return_type = get_return_type(support_dict=return_type_name_or_dict)
        ast.add_path(
            path=method, framework_reference=sympc, return_type_name=return_type
        )

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()
    return ast

