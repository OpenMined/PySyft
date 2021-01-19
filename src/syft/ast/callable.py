# stdlib
from types import ModuleType
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# syft relative
from .. import ast
from .. import lib
from ..core.node.common.action.function_or_constructor_action import (
    RunFunctionOrConstructorAction,
)


class Callable(ast.attribute.Attribute):
    """A method, function, or constructor which can be directly executed"""

    def __init__(
        self,
        path_and_name: Optional[str] = None,
        object_ref: Optional[Any] = None,
        return_type_name: Optional[str] = None,
        client: Optional[Any] = None,
        is_static: Optional[bool] = False,
    ):
        super().__init__(
            path_and_name=path_and_name,
            object_ref=object_ref,
            return_type_name=return_type_name,
            client=client,
        )

        self.is_static = is_static

    def __call__(
        self,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Optional[Union["Callable", CallableT]]:
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
                args=downcast_args, kwargs=downcast_kwargs, client=self.client
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
                return ptr

        path = kwargs["path"]
        index = kwargs["index"]

        if len(path) == index:
            return self.object_ref
        else:
            return self.attrs[path[index]](path=path, index=index + 1)

    def add_path(
        self,
        path: List[str],
        index: int,
        return_type_name: Optional[str] = None,
    ) -> None:
        if index >= len(path) or path[index] in self.attrs:
            return

        attr_ref = getattr(self.object_ref, path[index])

        if isinstance(attr_ref, ModuleType):
            raise Exception("Module cannot be attr of callable.")

        self.attrs[path[index]] = ast.callable.Callable(
            path_and_name=".".join(path[: index + 1]),
            object_ref=attr_ref,
            return_type_name=return_type_name,
            client=self.client,
        )
