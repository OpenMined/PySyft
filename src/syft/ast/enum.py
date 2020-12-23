# stdlib
from types import ModuleType
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# syft relative
from .. import ast
from ..core.common.pointer import AbstractPointer
from ..core.node.common.action.get_enum_attribute_action import EnumAttributeAction


class EnumAttribute(ast.attribute.Attribute):
    def __init__(
        self,
        parent: ast.attribute.Attribute,
        path_and_name: Optional[str] = None,
        return_type_name: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.parent = parent
        super().__init__(
            path_and_name=path_and_name,
            return_type_name=return_type_name,
            client=client,
        )

    def get_remote_enum_attribute(self) -> AbstractPointer:
        if self.path_and_name is None:
            raise ValueError("MAKE PROPER SCHEMA - Can't get enum attribute")

        if self.client is None:
            raise ValueError(
                "MAKE PROPER SCHEMA - Can't get remote value if there is no remote "
                "client"
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

    def solve_get_enum_attribute(self) -> int:
        return getattr(self.parent.object_ref, self.name)

    def __call__(
        self,
        path: Union[List[str], str],
        index: int = 0,
        obj_type: Optional[type] = None,
    ) -> None:
        raise ValueError("MAKE PROPER SCHEMA, THIS SHOULD NEVER BE CALLED")

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[ModuleType] = None,
        is_static: bool = False,
    ) -> None:
        raise ValueError("MAKE PROPER SCHEMA")
