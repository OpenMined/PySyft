from .. import ast
from ..core.node.common.action.function_or_constructor_action import (
    RunFunctionOrConstructorAction,
)
from .util import module_type, unsplit


class Callable(ast.attribute.Attribute):

    """A method, function, or constructor which can be directly executed"""

    def __call__(self, *args, return_callable=False, **kwargs):

        if self.client is not None and return_callable is False:
            print(f"call {self.path_and_name} on client {self.client}")

            path_and_name = "torch.zeros"
            args = [2, 3]
            kwargs = {}

            return_tensor_type_pointer_type = self.client.lib_ast(
                path=self.return_type_name, return_callable=True
            ).pointer_type
            ptr = return_tensor_type_pointer_type(location=self.client)

            msg = RunFunctionOrConstructorAction(
                path=path_and_name,
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
            return self.ref
        else:
            return self.attrs[path[index]](
                path, index + 1, return_callable=return_callable
            )

    def add_path(self, path, index, return_type_name=None):

        self.return_type_name = return_type_name

        if index < len(path):
            if path[index] not in self.attrs:

                attr_ref = getattr(self.ref, path[index])

                if isinstance(attr_ref, module_type):
                    raise Exception("Module cannot be attr of callable.")
                else:
                    self.attrs[path[index]] = ast.method.Method(
                        name=path[index],
                        path_and_name=unsplit(path[: index + 1]),
                        ref=attr_ref,
                        return_type_name=return_type_name,
                    )
