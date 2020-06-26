from .. import ast

from .util import unsplit
from .util import module_type
from .util import class_type
from .util import func_type
from .util import builtin_func_type


class Module(ast.attribute.Attribute):

    """A module which contains other modules or callables."""

    def add_attr(self, attr_name, attr):
        self.attrs[attr_name] = attr

    def __call__(self, path=None, index=0):
        if isinstance(path, str):
            path = path.split(".")
        return self.attrs[path[index]](path, index + 1)

    def add_path(self, path, index):

        if path[index] not in self.attrs:

            attr_ref = getattr(self.ref, path[index])

            if isinstance(attr_ref, module_type):
                self.attrs[path[index]] = Module(
                    path[index], unsplit(path[: index + 1]), attr_ref
                )
            elif isinstance(attr_ref, class_type):
                self.attrs[path[index]] = Class(
                    path[index], unsplit(path[: index + 1]), attr_ref
                )
            elif isinstance(attr_ref, func_type):
                self.attrs[path[index]] = Function(
                    path[index], unsplit(path[: index + 1]), attr_ref
                )
            elif isinstance(attr_ref, builtin_func_type):
                self.attrs[path[index]] = Function(
                    path[index], unsplit(path[: index + 1]), attr_ref
                )

        self.attrs[path[index]].add_path(path, index + 1)
