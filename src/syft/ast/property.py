# stdlib
from typing import Any
from typing import Optional

# syft relative
from .. import ast
from ..core.node.common.action.function_or_constructor_action import (
    RunFunctionOrConstructorAction,
)


class Property(ast.attribute.Attribute):
    client: Optional[Any]

    def __init__(
        self,
        name: Optional[str] = None,
        path_and_name: Optional[str] = None,
        ref: Optional[Any] = None,
        return_type_name: Optional[str] = None,
    ):
        super().__init__(name, path_and_name, ref, return_type_name)
