# stdlib
import functools
from typing import Any
from typing import Dict
from typing import Union

# third party
from packaging import version
import transformers

# syft relative
from . import batchencoding  # noqa: 401
from . import model_config  # noqa: 401
from . import tokenizer  # noqa: 401
from ...ast.globals import Globals
from ...logger import info

# from .allowlist import allowlist
from ..util import generic_update_ast
from .allowlist import allowlist

# The library name
LIB_NAME = "transformers"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
}

TRANSFORMERS_VERSION = version.parse(transformers.__version__.split("+")[0])


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    if isinstance(support_dict, str):
        return True
    else:
        # if we are on either side of the min or max versions we don't support this op
        if "min_version" in support_dict and TRANSFORMERS_VERSION < version.parse(
            support_dict["min_version"]
        ):
            return False
        if "max_version" in support_dict and TRANSFORMERS_VERSION > version.parse(
            support_dict["max_version"]
        ):
            return False
        return True


def create_ast(client: Any = None) -> Globals:
    ast = Globals(client)

    # most methods work in all versions and have a single return type
    # for the more complicated ones we pass a dict with keys like return_type and
    # min_version
    for method, return_type_name_or_dict in allowlist.items():
        if version_supported(support_dict=return_type_name_or_dict):
            return_type = get_return_type(support_dict=return_type_name_or_dict)
            if return_type == "unknown":
                # this allows us to import them for testing
                continue
            ast.add_path(
                path=method,
                framework_reference=transformers,
                return_type_name=return_type,
            )

        else:
            info(f"Skipping {method} not supported in {TRANSFORMERS_VERSION}")

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.store_init_args()
        klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
