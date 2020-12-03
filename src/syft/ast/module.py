# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# syft relative
from .. import ast
from ..ast.callable import Callable
from .util import builtin_func_type
from .util import class_type
from .util import func_type
from .util import module_type
from .util import unsplit


class Module(ast.attribute.Attribute):

    """A module which contains other modules or callables."""

    lookup_cache: Dict[Any, Any] = {}

    def add_attr(
        self,
        attr_name: str,
        attr: Optional[Union[Callable, CallableT]],
    ) -> None:
        self.__setattr__(attr_name, attr)
        if attr is not None:
            self.attrs[attr_name] = attr

            # if add_attr is called directly we need to cache the path as well
            attr_ref = getattr(attr, "ref", None)
            path = getattr(attr, "path_and_name", None)
            if attr_ref not in self.lookup_cache and path is not None:
                self.lookup_cache[attr_ref] = path

    def __call__(
        self,
        path: Union[str, List[str]] = [],
        index: int = 0,
        return_callable: bool = False,
        obj_type: Optional[type] = None,
    ) -> Optional[Union[Callable, CallableT]]:
        if obj_type is not None:
            if obj_type in self.lookup_cache:
                path = self.lookup_cache[obj_type]

        if isinstance(path, str):
            path = path.split(".")

        resolved = self.attrs[path[index]](
            path=path,
            index=index + 1,
            return_callable=return_callable,
            obj_type=obj_type,
        )
        return resolved

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
                klass = ast.klass.Class(
                    name=path[index],
                    path_and_name=unsplit(path[: index + 1]),
                    ref=attr_ref,
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
        attr_ref = getattr(self.ref, path[index], None)
        if attr_ref is not None and attr_ref not in self.lookup_cache:
            self.lookup_cache[attr_ref] = path
        if hasattr(attr, "add_path"):
            attr.add_path(  # type: ignore
                path=path, index=index + 1, return_type_name=return_type_name
            )
