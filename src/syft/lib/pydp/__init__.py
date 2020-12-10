# stdlib
from typing import Any as TypeAny
from typing import Dict
from typing import Union

# third party
from packaging import version
import pydp

# syft relative
from ...ast.globals import Globals
from ...ast.klass import Class
from ...ast.module import Module
from .allowlist import allowlist


def get_parent(path: str, root: TypeAny) -> Module:
    parent = root
    for step in path.split(".")[:-1]:
        parent = parent.attrs[step]
    return parent


PYDP_VERSION = version.parse(pydp.__version__)


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    if isinstance(support_dict, str):
        return True
    else:
        return PYDP_VERSION >= version.parse(support_dict["min_version"])


def create_pydp_ast() -> Globals:
    ast = Globals()

    # most methods work in all versions and have a single return type
    # for the more complicated ones we pass a dict with keys like return_type and
    # min_version
    for method, return_type_name_or_dict in allowlist.items():
        if version_supported(support_dict=return_type_name_or_dict):
            return_type = get_return_type(support_dict=return_type_name_or_dict)
            ast.add_path(
                path=method, framework_reference=pydp, return_type_name=return_type,
            )
        else:
            print(f"Skipping pydp.{method} not supported in {PYDP_VERSION}")

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()
    return ast
