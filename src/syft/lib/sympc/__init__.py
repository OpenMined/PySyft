# stdlib
from typing import Union
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import sympc

# syft relative
from . import session  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals


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

    return ast
