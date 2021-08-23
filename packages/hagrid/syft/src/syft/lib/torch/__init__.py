# stdlib
from typing import Any
from typing import Dict
from typing import Union

# third party
from packaging import version
import torch

# relative
from . import device  # noqa: 401
from . import parameter  # noqa: 401
from . import return_types  # noqa: 401
from . import size  # noqa: 401
from . import uppercase_tensor  # noqa: 401
from ...ast import add_dynamic_objects
from ...ast.globals import Globals
from ...logger import info
from .allowlist import allowlist
from .allowlist import dynamic_allowlist

TORCH_VERSION = version.parse(torch.__version__.split("+")[0])


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
        if "min_version" in support_dict and TORCH_VERSION < version.parse(
            support_dict["min_version"]
        ):
            return False
        if "max_version" in support_dict and TORCH_VERSION > version.parse(
            support_dict["max_version"]
        ):
            return False
        return True


def create_torch_ast(client: Any = None) -> Globals:
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
                path=method, framework_reference=torch, return_type_name=return_type
            )
            # add all the torch.nn.Parameter hooks
            if method.startswith("torch.Tensor."):
                method = method.replace("torch.Tensor.", "torch.nn.Parameter.")
                return_type = return_type.replace("torch.Tensor", "torch.nn.Parameter")
                ast.add_path(
                    path=method, framework_reference=torch, return_type_name=return_type
                )
        else:
            info(f"Skipping {method} not supported in {TORCH_VERSION}")

    add_dynamic_objects(ast, list(dynamic_allowlist.items()))

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast
