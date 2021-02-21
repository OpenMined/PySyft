# stdlib
import inspect
import sys
from types import ModuleType
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Union

# syft relative
from .. import ast
from ..ast.callable import Callable
from ..logger import traceback_and_raise


def is_static_method(host_object, attr):  # type: ignore
    value = getattr(host_object, attr)

    if not hasattr(host_object, "__mro__"):
        return False

    for cls in inspect.getmro(host_object):
        if inspect.isroutine(value):
            if attr in cls.__dict__:
                bound_value = cls.__dict__[attr]
                if isinstance(bound_value, staticmethod):
                    return True
    return False


class Module(ast.attribute.Attribute):

    """A module which contains other modules or callables."""

    def __init__(
        self,
        client: Optional[Any],
        object_ref: Union[CallableT, ModuleType],
        parent: Optional[ast.attribute.Attribute] = None,
        path_and_name: Optional[str] = None,
        return_type_name: Optional[str] = None,
    ):
        super().__init__(
            path_and_name=path_and_name,
            object_ref=object_ref,
            return_type_name=return_type_name,
            client=client,
            parent=parent,
        )

        if object_ref is None and self.name:
            try:
                self.object_ref = sys.modules[path_and_name]
            except Exception as e:
                self.object_ref = getattr(self._parent.object_ref, self.name)

    def add_attr(
        self,
        attr_name: str,
        attr: Optional[Union[Callable, CallableT]],
        is_static: bool = False,
    ) -> None:
        self.__setattr__(attr_name, attr)

        if is_static is True:
            traceback_and_raise(
                ValueError("Static methods shouldn't be added to an object.")
            )

        if attr is None:
            traceback_and_raise(ValueError("An attribute reference has to be passed."))

        # if add_attr is called directly we need to cache the path as well
        attr_ref = getattr(attr, "object_ref", None)
        path = getattr(attr, "path_and_name", None)
        if attr_ref not in self.lookup_cache and path is not None:
            self.lookup_cache[attr_ref] = path

        self.attrs[attr_name] = attr  # type: ignore

    def __call__(
        self,
        path: Union[List[str], str],
        index: int = 0,
        obj_type: Optional[type] = None,
    ) -> Optional[Union[Callable, CallableT]]:

        self.apply_node_changes()

        if obj_type is not None:
            if obj_type in self.lookup_cache:
                path = self.lookup_cache[obj_type]

        _path: List[str] = (
            path.split(".") if isinstance(path, str) else path if path else []
        )

        if not _path:
            traceback_and_raise(
                ValueError("Can't execute remote call if path is not specified.")
            )

        resolved = self.attrs[_path[index]](
            path=_path, index=index + 1, obj_type=obj_type
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
        index: int = 0,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[ModuleType] = None,
        is_static: bool = False,
    ) -> None:
        if index >= len(path):
            return

        if path[index] not in self.attrs:
            attr_ref = getattr(self.object_ref, path[index])

            if inspect.ismodule(attr_ref):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.module.Module(
                        path_and_name=".".join(path[: index + 1]),
                        object_ref=attr_ref,
                        return_type_name=return_type_name,
                        client=self.client,
                        parent=self,
                    ),
                )
            elif inspect.isclass(attr_ref):
                klass = ast.klass.Class(
                    path_and_name=".".join(path[: index + 1]),
                    object_ref=attr_ref,
                    return_type_name=return_type_name,
                    client=self.client,
                    parent=self,
                )
                self.add_attr(
                    attr_name=path[index],
                    attr=klass,
                )
            elif inspect.isfunction(attr_ref) or inspect.isbuiltin(attr_ref):
                is_static = is_static_method(self.object_ref, path[index])  # type: ignore

                self.add_attr(
                    attr_name=path[index],
                    attr=ast.callable.Callable(
                        path_and_name=".".join(path[: index + 1]),
                        object_ref=attr_ref,
                        return_type_name=return_type_name,
                        client=self.client,
                        is_static=is_static,
                        parent=self,
                    ),
                )
            elif inspect.isdatadescriptor(attr_ref):
                self.add_attr(
                    attr_name=path[index],
                    attr=ast.property.Property(
                        path_and_name=".".join(path[: index + 1]),
                        object_ref=attr_ref,
                        return_type_name=return_type_name,
                        client=self.client,
                        parent=self,
                    ),
                )
            elif index == len(path) - 1:
                static_attribute = ast.static_attr.StaticAttribute(
                    path_and_name=".".join(path[: index + 1]),
                    return_type_name=return_type_name,
                    client=self.client,
                    parent=self,
                )
                setattr(self, path[index], static_attribute)
                self.attrs[path[index]] = static_attribute
                return

        attr = self.attrs[path[index]]
        attr_ref = getattr(self.object_ref, path[index], None)
        if attr_ref is not None and attr_ref not in self.lookup_cache:
            self.lookup_cache[attr_ref] = path

        attr.add_path(path=path, index=index + 1, return_type_name=return_type_name)

    def __getattribute__(self, item: str) -> Any:
        # super().apply_node_changes()

        target_object = super().__getattribute__(item)
        if isinstance(target_object, ast.static_attr.StaticAttribute):
            return target_object.get_remote_value()
        return target_object

    def __setattr__(self, key: str, value: Any) -> None:
        # self.apply_node_changes()

        if hasattr(super(), "attrs"):
            attrs = super().__getattribute__("attrs")
            if key in attrs:
                target_object = self.attrs[key]
                if isinstance(target_object, ast.static_attr.StaticAttribute):
                    return target_object.set_remote_value(value)

        return super().__setattr__(key, value)

    def fetch_live_object(self) -> object:
        try:
            return sys.modules[self.path_and_name]
        except Exception as e:
            return getattr(self._parent.object_ref, self.name)
