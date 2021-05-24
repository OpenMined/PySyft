# stdlib
from typing import Any
from typing import Dict
from typing import Union

# third party
from packaging import version
import torchvision as tv

# syft relative
from ...ast.globals import Globals
from ...logger import critical
from .allowlist import allowlist

TORCHVISION_VERSION = version.parse(tv.__version__)


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    if isinstance(support_dict, str):
        return True
    else:
        if "min_version" not in support_dict.keys():
            return True
        return TORCHVISION_VERSION >= version.parse(support_dict["min_version"])


def create_torchvision_ast(client: Any = None) -> Globals:
    ast = Globals(client)

    # most methods work in all versions and have a single return type
    # for the more complicated ones we pass a dict with keys like return_type and
    # min_version
    for method, return_type_name_or_dict in allowlist.items():
        if version_supported(support_dict=return_type_name_or_dict):
            return_type = get_return_type(support_dict=return_type_name_or_dict)
            ast.add_path(
                path=method,
                framework_reference=tv,
                return_type_name=return_type,
            )
        else:
            critical(
                f"Skipping torchvision.{method} not supported in {TORCHVISION_VERSION}"
            )

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()
    return ast
