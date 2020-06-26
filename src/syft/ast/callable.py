from .. import ast

from .util import unsplit
from .util import module_type


class Callable(ast.attribute.Attribute):

    """A method, function, or constructor which can be directly executed"""

    def __call__(self, path, index):
        if len(path) == index:
            return self.ref
        else:
            return self.attrs[path[index]](path, index + 1)

    def add_path(self, path, index):
        if index < len(path):
            if path[index] not in self.attrs:

                attr_ref = getattr(self.ref, path[index])

                if isinstance(attr_ref, module_type):
                    raise Exception("Module cannot be attr of callable.")
                else:
                    self.attrs[path[index]] = sy.ast.Method(
                        path[index], unsplit(path[: index + 1]), attr_ref
                    )
