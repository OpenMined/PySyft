# stdlib
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Union

# syft relative
from .. import ast
from ..ast.callable import Callable
from ..lib.generic import ObjectConstructor
from .util import builtin_func_type
from .util import class_type
from .util import func_type
from .util import module_type
from .util import unsplit


class Module(ast.attribute.Attribute):

    """A module which contains other modules or callables."""

    def add_attr(
        self,
        attr_name: str,
        attr: Optional[Union[Callable, CallableT]],
    ) -> None:
        self.__setattr__(attr_name, attr)
        if attr is not None:
            self.attrs[attr_name] = attr

    def __call__(
        self,
        path: Union[str, List[str]] = [],
        index: int = 0,
        return_callable: bool = False,
    ) -> Optional[Union[Callable, CallableT]]:
        if isinstance(path, str):
            path = path.split(".")
        return self.attrs[path[index]](
            path=path, index=index + 1, return_callable=return_callable
        )

    def __repr__(self) -> str:
        out = "Module:\n"
        for name, module in self.attrs.items():
            out += "\t." + name + " -> " + str(module).replace("\t.", "\t\t.") + "\n"

        return out

    def add_path(
        self,
        path: List[str],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[Union[Callable, CallableT]] = None,
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
                # call the ClassFactory now so that the type can be subclassed later by
                # end users, see klass.py.
                klass = ast.klass.ClassFactory(
                    name=path[index],
                    path_and_name=unsplit(path[: index + 1]),
                    ref=attr_ref,
                    return_type_name=return_type_name,
                )
                self.add_attr(
                    attr_name=path[index],
                    attr=klass,
                )
            elif isinstance(attr_ref, ObjectConstructor):
                # call the ClassFactory now so that the type can be subclassed later by
                # end users, see klass.py.
                klass = ast.klass.ClassFactory(
                    name=path[index],
                    path_and_name=unsplit(path[: index + 1]),
                    ref=attr_ref.original_type,  # type: ignore
                    return_type_name=return_type_name,
                )
                self.add_attr(
                    attr_name=path[index],
                    attr=klass,
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

        attr = self.attrs[path[index]]
        if hasattr(attr, "add_path"):
            attr.add_path(  # type: ignore
                path=path, index=index + 1, return_type_name=return_type_name
            )
