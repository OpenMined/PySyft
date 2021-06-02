# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import Optional
from typing import Tuple
from typing import Union

# syft relative
from .. import ast
from ..logger import traceback_and_raise


class Property(ast.attribute.Attribute):
    """Creates property objects which implements getter and setter objects.
    Each valid action on AST triggers GetSetPropertyAction.
    """

    def __init__(
        self,
        path_and_name: str,
        parent: ast.attribute.Attribute,
        object_ref: Optional[Any] = None,
        return_type_name: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        super().__init__(
            path_and_name=path_and_name,
            parent=parent,
            object_ref=object_ref,
            return_type_name=return_type_name,
            client=client,
        )

        self.is_static = False

    def __call__(
        self,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Optional[Union[Any, CallableT]]:
        traceback_and_raise(ValueError("Property should never be called."))
