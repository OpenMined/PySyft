"""This module contains `StaticAttribute`, an AST node representing a method,
 function, or constructor which can be directly executed."""

# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import Optional
from typing import Union

# relative
from .. import ast
from .. import lib
from ..core.common.pointer import AbstractPointer
from ..core.common.uid import UID
from ..core.node.common.action.get_or_set_static_attribute_action import (
    GetSetStaticAttributeAction,
)
from ..core.node.common.action.get_or_set_static_attribute_action import (
    StaticAttributeAction,
)
from ..logger import traceback_and_raise


class StaticAttribute(ast.attribute.Attribute):
    """A method, function, or constructor which can be directly executed."""

    def __init__(
        self,
        parent: ast.attribute.Attribute,
        path_and_name: str,
        return_type_name: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        """Base constructor.

        Args:
            parent: The parent node is needed.
            path_and_name: The path for the current node, e.g. `syft.lib.python.List`.
            return_type_name: The return type name of given action as a string with it's full path.
            client: The client for which all computation is being executed.
        """
        super().__init__(
            path_and_name=path_and_name,
            return_type_name=return_type_name,
            client=client,
            parent=parent,
        )

    def get_remote_value(self) -> AbstractPointer:
        """Remote execution is performed when AST is constructed with a client.

        The get_remote_value method triggers GetSetStaticAttributeAction on the AST.

        Returns:
            AbstractPointer: Pointer to remote value.
        """
        if self.path_and_name is None:
            traceback_and_raise(
                ValueError("Can't execute remote get if path is not specified.")
            )

        if self.client is None:
            if self.client is None:
                traceback_and_raise(
                    ValueError(
                        "Can't get remote enum attribute if there is no client"
                        "set to get it from"
                    )
                )

        return_tensor_type_pointer_type = self.client.lib_ast.query(
            path=self.return_type_name
        ).pointer_type

        ptr = return_tensor_type_pointer_type(client=self.client)

        msg = GetSetStaticAttributeAction(
            path=self.path_and_name,
            id_at_location=ptr.id_at_location,
            address=self.client.address,
            action=StaticAttributeAction.GET,
        )
        self.client.send_immediate_msg_without_reply(msg=msg)
        return ptr

    def solve_get_value(self) -> Any:
        """Local execution of the getter function is performed.

        The `solve_get_value` method executes the getter function on the AST.

        Raises:
            ValueError : If `path_and_name` is `None`.

        Returns:
            Value of the AST node
        """
        self.apply_node_changes()

        if self.path_and_name is None:
            raise ValueError("path_and_name should not be None")

        return getattr(self.parent.object_ref, self.path_and_name.rsplit(".")[-1])

    def solve_set_value(self, set_value: Any) -> None:
        """Local execution of setter function is performed.

        The `solve_set_value` method executes the setter function on the AST.

        Args:
            set_value: The value to set to.

        Raises:
            ValueError : If `path_and_name` is `None`.

        """
        self.apply_node_changes()

        if self.path_and_name is None:
            raise ValueError("path_and_none should not be None")

        setattr(self.parent.object_ref, self.path_and_name.rsplit(".")[-1], set_value)

    def set_remote_value(self, set_arg: Any) -> Any:
        """Remote execution of setter function is performed when AST is constructed with a client.

        The set_remote_value method triggers GetSetStaticAttributeAction on the AST.

        Args:
            set_arg: The value to set to.

        Raises:
            ValueError: If `client` is `None` or `path_and_name` is `None`.

        Returns:
            Pointer to the object
        """
        if self.client is None:
            raise ValueError(
                "MAKE PROPER SCHEMA - Can't get remote value if there is no remote "
                "client"
            )

        if self.path_and_name is None:
            raise ValueError("MAKE PROPER SCHEMA")

        resolved_pointer_type = self.client.lib_ast.query(self.return_type_name)
        result = resolved_pointer_type.pointer_type(client=self.client)
        result_id_at_location: Optional[UID] = getattr(result, "id_at_location", None)
        if result_id_at_location is None:
            raise Exception("Can't get remote value if there is no id_at_location")

        downcasted_set_arg = lib.python.util.downcast(set_arg)
        downcasted_set_arg_ptr = downcasted_set_arg.send(self.client)

        cmd = GetSetStaticAttributeAction(
            path=self.path_and_name,
            id_at_location=result_id_at_location,
            address=self.client.address,
            action=StaticAttributeAction.SET,
            set_arg=downcasted_set_arg_ptr,
        )
        self.client.send_immediate_msg_without_reply(msg=cmd)
        return result

    def __call__(  # type: ignore
        self, action: StaticAttributeAction
    ) -> Optional[Union["ast.callable.Callable", CallableT]]:
        """A `StaticAttribute` attribute is not callable.

        Args:
            action: `GET` or `SET` action

        Raises:
            ValueError: If the function is called.
        """
        raise ValueError("MAKE PROPER SCHEMA, THIS SHOULD NEVER BE CALLED")

    def add_path(self, *args: Any, **kwargs: Any) -> None:  # type: ignore
        """An `StaticAttribute` can no longer have children nodes.

        Args:
            *args: List of arguments.
            **kwargs: Dict of Keyword arguments.

        Raises:
            ValueError: If the function is called.

        """
        raise ValueError("MAKE PROPER SCHEMA")
