"""This module contains `Callable`, an AST node representing a method (can be static),
global function, or constructor which can be directly executed."""

# stdlib
from types import ModuleType
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# relative
from .. import ast
from .. import lib
from ..core.node.abstract.node import AbstractNodeClient
from ..core.node.common.action.function_or_constructor_action import (
    RunFunctionOrConstructorAction,
)
from ..logger import traceback_and_raise
from ..util import inherit_tags
from .util import module_type


class Callable(ast.attribute.Attribute):
    """Represent a method (can be static), global function, or constructor which can be directly executed."""

    def __init__(
        self,
        path_and_name: str,
        parent: ast.attribute.Attribute,
        object_ref: Optional[Any] = None,
        return_type_name: Optional[str] = None,
        client: Optional[AbstractNodeClient] = None,
        is_static: Optional[bool] = False,
    ):
        """Base constructor for Callable.

        Args:
            path_and_name: The path for the current node, e.g. `syft.lib.python.List`.
            parent: The parent node and it's attributes in the AST.
            object_ref: The actual python object for which the computation is being made.
            return_type_name: The return type name of the given action as a string with its full path.
            client: The client for which all computation is being executed.
            is_static: If True, the object has to be resolved on the AST, not on an existing pointer.
        """
        super().__init__(
            path_and_name=path_and_name,
            object_ref=object_ref,
            return_type_name=return_type_name,
            client=client,
            parent=parent,
        )

        self.is_static = is_static

    def __call__(
        self,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Optional[Union["Callable", CallableT]]:
        """The `__call__` method on a `Callable` has two possible roles, e.g.

        1. If the client is set, execute the function for the client and return the appropriate pointer
        given the `return_type_name`.

        2. If the client is not set, then the `__call__` is used as a query on the ast.

        Args:
            *args: arguments of `callable`
            **kwargs: keyword arguments of `callable`

        Returns:
            If client is not set, returns `callable` node in AST at given path.
        """
        self.apply_node_changes()

        if self.client is not None:
            return_tensor_type_pointer_type = self.client.lib_ast.query(
                path=self.return_type_name
            ).pointer_type

            ptr = return_tensor_type_pointer_type(client=self.client)

            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=args, kwargs=kwargs)

            # then we convert anything which isn't a pointer into a pointer
            pointer_args, pointer_kwargs = ast.klass.pointerize_args_and_kwargs(
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=self.client,
                gc_enabled=False,
            )

            if self.path_and_name is not None:
                msg = RunFunctionOrConstructorAction(
                    path=self.path_and_name,
                    args=pointer_args,
                    kwargs=pointer_kwargs,
                    id_at_location=ptr.id_at_location,
                    address=self.client.address,
                    is_static=self.is_static,
                )

                self.client.send_immediate_msg_without_reply(msg=msg)

                inherit_tags(
                    attr_path_and_name=self.path_and_name,
                    result=ptr,
                    self_obj=None,
                    args=args,
                    kwargs=kwargs,
                )
                return ptr

        if "path" not in kwargs or "index" not in kwargs:
            traceback_and_raise(
                ValueError(
                    "AST with no client attached tries to execute remote function."
                )
            )
        path = kwargs["path"]
        index = kwargs["index"]

        if len(path) == index:
            return self.object_ref
        else:
            return self.attrs[path[index]](path=path, index=index + 1)

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[ModuleType] = None,
        is_static: bool = False,
    ) -> None:
        """The add_path adds new nodes in AST based on type of current node and type of object to be added.

        Args:
            path: The node path added in AST, e.g. `syft.lib.python.List` or ["syft", "lib", "python", "List].
            index: The associated position in the path for the current node.
            return_type_name: The return type name of the given action as a string with its full path.
            framework_reference: The Python framework in which we can solve the same path to obtain the Python object.
            is_static: If the node represents a static method.
        """
        if index >= len(path) or path[index] in self.attrs:
            return

        attr_ref = getattr(self.object_ref, path[index])

        if isinstance(attr_ref, module_type):
            traceback_and_raise(
                ValueError("Module cannot be an attribute of Callable.")
            )

        self.attrs[path[index]] = ast.callable.Callable(
            path_and_name=".".join(path[: index + 1]),
            object_ref=attr_ref,
            return_type_name=return_type_name,
            client=self.client,
            parent=self,
        )
