"""This module contains `Property` attribute representing property objects which
implements getter and setter objects."""

# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import Optional
from typing import Tuple
from typing import Union

# relative
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
    ) -> None:
        """Base constructor for Property Attribute.

        Args:
            client: The client for which all computation is being executed.
            path_and_name: The path for the current node, e.g. `syft.lib.python.List`.
            object_ref: The actual python object for which the computation is being made.
            return_type_name: The given action's return type name, with its full path, in string format.
            parent: The parent node in the AST.
        """
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
        """`Property` attribute is not callable.

        Args:
            *args: List of arguments.
            **kwargs: Keyword arguments.

        Raises:
            ValueError: If the function is called.
        """
        traceback_and_raise(ValueError("Property should never be called."))
