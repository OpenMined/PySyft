from .. import ast

from .util import unsplit
from .util import module_type


class Callable(ast.attribute.Attribute):

    """A method, function, or constructor which can be directly executed"""

    def __call__(self, path, index, return_callable=False):
        if len(path) == index:
            if(return_callable):
                return self
            return self.ref
        else:
            return self.attrs[path[index]](path, index + 1, return_callable=return_callable)

    def add_path(self, path, index, return_type_name=None):

        self.return_type_name = return_type_name

        if index < len(path):
            if path[index] not in self.attrs:

                attr_ref = getattr(self.ref, path[index])

                if isinstance(attr_ref, module_type):
                    raise Exception("Module cannot be attr of callable.")
                else:
                    self.attrs[path[index]] = ast.method.Method(
                        name=path[index], path_and_name=unsplit(path[: index + 1]), ref=attr_ref, return_type_name=return_type_name
                    )
