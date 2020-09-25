# stdlib
from typing import Dict
from typing import Union

# third party
from packaging import version
import torch

# syft relative
from . import parameter  # noqa: 401
from . import uppercase_tensor  # noqa: 401
from ...ast.globals import Globals
from .allowlist import allowlist

TORCH_VERSION = version.parse(torch.__version__)


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    if isinstance(support_dict, str):
        return True
    else:
        return TORCH_VERSION >= version.parse(support_dict["min_version"])


def create_torch_ast() -> Globals:
    ast = Globals()

    # most methods work in all versions and have a single return type
    # for the more complicated ones we pass a dict with keys like return_type and
    # min_version
    for method, return_type_name_or_dict in allowlist.items():
        if version_supported(support_dict=return_type_name_or_dict):
            return_type = get_return_type(support_dict=return_type_name_or_dict)
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
            print(f"Skipping torch.{method} not supported in {TORCH_VERSION}")

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()
    return ast
