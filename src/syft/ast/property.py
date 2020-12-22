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

    def __call__(self):
        return_tensor_type_pointer_type = self.client.lib_ast.query(
            path=self.return_type_name
        ).pointer_type

        ptr = return_tensor_type_pointer_type(client=self.client)
        if self.path_and_name is not None:
            msg = RunClassProperty(
                path=self.path_and_name,
                args=tuple(),
                kwargs=dict(),
                id_at_location=ptr.id_at_location,
                address=self.client.address,
            )
            self.client.send_immediate_msg_without_reply(msg=msg)
        return ptr
