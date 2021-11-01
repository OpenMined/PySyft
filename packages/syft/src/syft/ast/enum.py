"""This module contains `EnumAttribute`, an AST node representing the
 attributes of a Python Enum which are only gettable, not settable."""

# stdlib
from enum import Enum
from typing import Any
from typing import Optional

# relative
from .. import ast
from ..core.common.pointer import AbstractPointer
from ..core.node.abstract.node import AbstractNodeClient
from ..core.node.common.action.get_enum_attribute_action import EnumAttributeAction
from ..logger import traceback_and_raise


class EnumAttribute(ast.attribute.Attribute):
    """An EnumAttribute represents the attributes of a Python Enum which are only gettable, not settable."""

    def __init__(
        self,
        parent: ast.attribute.Attribute,
        path_and_name: str,
        return_type_name: Optional[str] = None,
        client: Optional[AbstractNodeClient] = None,
    ) -> None:
        """Base constructor for Enum Attribute.

        Args:
            parent: The parent node is needed when solving `EnumAttributes`.
            path_and_name: The path for the current node, e.g. `syft.lib.python.List`.
            return_type_name: The return type name of given action as a string with its full path.
            client: The client for which all computation is being executed.
        """
        super().__init__(
            path_and_name=path_and_name,
            return_type_name=return_type_name,
            client=client,
            parent=parent,
        )

    def get_remote_enum_attribute(self) -> AbstractPointer:
        """Remote getter on an `Enum` attribute in the AST.

        Raises:
            ValueError: Cannot get Enum attribute.

        Returns:
            A pointer to the remote enum attribute.
        """
        if self.path_and_name is None:
            traceback_and_raise(
                ValueError(
                    "Can't get enum attribute, path_and_name to solve it "
                    "has not been set."
                )
            )

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

        msg = EnumAttributeAction(
            path=self.path_and_name,
            id_at_location=ptr.id_at_location,
            address=self.client.address,
        )
        self.client.send_immediate_msg_without_reply(msg=msg)
        return ptr

    def solve_get_enum_attribute(self) -> Enum:
        """Local getter on an `Enum` attribute in the AST.

        Raises:
            ValueError: Cannot get Enum attribute.

        Returns:
            The `Enum` object from the parent object reference.
        """
        if self.path_and_name is None:
            traceback_and_raise(
                ValueError(
                    "Can't get enum attribute, path_and_name to solve it "
                    "has not been set remotely."
                )
            )

        return getattr(self.parent.object_ref, self.name)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """An `Enum` attribute is not callable.

        Args:
            *args: Variable length argument list.
            *kwargs: keyword arguments.

        Raises:
            ValueError: If the function is called.
        """
        traceback_and_raise(
            ValueError("__call__ should never be executed on an enum " "attribute.")
        )

    def add_path(self, *args: Any, **kwargs: Any) -> None:
        """An `Enum` can no longer have children nodes.

        Args:
            *args: Variable length argument list.
            *kwargs: keyword arguments..

        Raises:
            ValueError: If the function is called.
        """
        traceback_and_raise(
            "__add__path should never be called on an enum attribute, "
            "enum attributes are leaf nodes in the AST."
        )
