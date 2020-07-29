from .. import ast

from .util import unsplit
from .util import module_type
from .util import class_type
from .util import func_type
from .util import builtin_func_type


class Module(ast.attribute.Attribute):

    """A module which contains other modules or callables."""

    def add_attr(self, attr_name, attr):
        self.__setattr__(attr_name, attr)
        self.attrs[attr_name] = attr

    def __call__(self, path=None, index=0, return_callable=False):
        if isinstance(path, str):
            path = path.split(".")
        return self.attrs[path[index]](path, index + 1, return_callable=return_callable)

    def add_path(self, path, index, return_type_name):

        if path[index] not in self.attrs:

            attr_ref = getattr(self.ref, path[index])

            if isinstance(attr_ref, module_type):
                self.add_attr(attr_name=path[index], attr=ast.module.Module(
                    path[index], unsplit(path[: index + 1]), attr_ref, return_type_name=return_type_name
                ))
            elif isinstance(attr_ref, class_type):
                self.add_attr(attr_name=path[index], attr=ast.klass.Class(
                    name=path[index], path_and_name=unsplit(path[: index + 1]), ref=attr_ref, return_type_name=return_type_name
                ))
            elif isinstance(attr_ref, func_type):
                self.add_attr(attr_name=path[index], attr=ast.function.Function(
                    path[index], unsplit(path[: index + 1]), attr_ref, return_type_name=return_type_name
                ))
            elif isinstance(attr_ref, builtin_func_type):
                self.add_attr(attr_name=path[index], attr=ast.function.Function(
                    path[index], unsplit(path[: index + 1]), attr_ref, return_type_name=return_type_name
                ))

        self.attrs[path[index]].add_path(path, index + 1, return_type_name=return_type_name)
