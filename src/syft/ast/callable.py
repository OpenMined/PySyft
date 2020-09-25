# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# syft relative
from .. import ast
from ..core.node.common.action.function_or_constructor_action import (
    RunFunctionOrConstructorAction,
)
from .util import module_type
from .util import unsplit


class Callable(ast.attribute.Attribute):
    client: Optional[Any]

    """A method, function, or constructor which can be directly executed"""

    def __call__(
        self,
        *args: Tuple[Any, ...],
        return_callable: bool = False,
        **kwargs: Any,
    ) -> Optional[Union["Callable", CallableT]]:
        print(
            "we are calling",
            type(self),
            args,
            kwargs,
            return_callable,
            self.return_type_name,
        )

        if (
            hasattr(self, "client")
            and self.client is not None
            and return_callable is False
        ):
            print("call ast with ", self.return_type_name, True)
            return_tensor_type_pointer_type = self.client.lib_ast(
                path=self.return_type_name, return_callable=True
            ).pointer_type

            print("this is our return tensor", return_tensor_type_pointer_type)
            ptr = return_tensor_type_pointer_type(client=self.client)

            if self.path_and_name is not None:
                msg = RunFunctionOrConstructorAction(
                    path=self.path_and_name,
                    args=args,
                    kwargs=kwargs,
                    id_at_location=ptr.id_at_location,
                    address=self.client.address,
                )

                self.client.send_immediate_msg_without_reply(msg=msg)
                return ptr

        path = kwargs["path"]
        index = kwargs["index"]

        if len(path) == index:
            if return_callable:
                return self
            print("we are here for", type(self), self.ref)
            return self.ref
        else:
            return self.attrs[path[index]](
                path=path, index=index + 1, return_callable=return_callable
            )

    def add_path(
        self, path: List[str], index: int, return_type_name: Optional[str] = None
    ) -> None:
        print("adding, path", type(self))

        if index < len(path):
            if path[index] not in self.attrs:

                attr_ref = getattr(self.ref, path[index])

                if isinstance(attr_ref, module_type):
                    raise Exception("Module cannot be attr of callable.")
                else:
                    is_property = type(attr_ref).__name__ == "getset_descriptor"

                    self.attrs[path[index]] = ast.method.Method(
                        name=path[index],
                        path_and_name=unsplit(path[: index + 1]),
                        ref=attr_ref,
                        return_type_name=return_type_name,
                        is_property=is_property,
                    )
        # elif index == len(path):
        #     print("DO WE GET HERE?")
        #     self.return_type_name = return_type_name
