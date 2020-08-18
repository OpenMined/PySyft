from typing import Optional, Union, List
from .. import ast
from .util import builtin_func_type, class_type, func_type, module_type, unsplit
from ..lib.generic import ObjectConstructor


class Module(ast.attribute.Attribute):

    """A module which contains other modules or callables."""

    def add_attr(self, attr_name, attr):
        self.__setattr__(attr_name, attr)
        self.attrs[attr_name] = attr

    def __call__(self, path=None, index=0, return_callable=False):
        if isinstance(path, str):
            path = path.split(".")
        return self.attrs[path[index]](
            path=path, index=index + 1, return_callable=return_callable
        )

    def __repr__(self):
        out = "Module:\n"
        for name, module in self.attrs.items():
            out += "\t." + name + " -> " + str(module).replace("\t.", "\t\t.") + "\n"

        return out

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[object] = None,
    ) -> None:
        if path[index] not in self.attrs:

            attr_ref = getattr(self.ref, path[index])

            if isinstance(attr_ref, module_type):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.module.Module(
                        path[index],
                        unsplit(path[: index + 1]),
                        attr_ref,
                        return_type_name=return_type_name,
                    ),
                )
            elif isinstance(attr_ref, class_type):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.klass.Class(
                        name=path[index],
                        path_and_name=unsplit(path[: index + 1]),
                        ref=attr_ref,
                        return_type_name=return_type_name,
                    ),
                )
            elif isinstance(attr_ref, ObjectConstructor):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.klass.Class(
                        name=path[index],
                        path_and_name=unsplit(path[: index + 1]),
                        ref=attr_ref.original_type,  # type: ignore
                        return_type_name=return_type_name,
                    ),
                )

            elif isinstance(attr_ref, func_type):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.function.Function(
                        path[index],
                        unsplit(path[: index + 1]),
                        attr_ref,
                        return_type_name=return_type_name,
                    ),
                )
            elif isinstance(attr_ref, builtin_func_type):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.function.Function(
                        path[index],
                        unsplit(path[: index + 1]),
                        attr_ref,
                        return_type_name=return_type_name,
                    ),
                )

        self.attrs[path[index]].add_path(
            path=path, index=index + 1, return_type_name=return_type_name
        )
