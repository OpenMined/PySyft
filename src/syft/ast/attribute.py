# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from types import ModuleType

# syft relative
from .. import ast


class Attribute:
    __slots__ = [
        "path_and_name",
        "object_ref",
        "attrs",
        "return_type_name",
        "client",
    ]

    lookup_cache: Dict[Any, Any] = {}

    def __init__(
        self,
        client: Optional[Any],
        path_and_name: Optional[str] = None,
        object_ref: Any = None,
        return_type_name: Optional[str] = None,
    ):
        self.path_and_name = path_and_name
        self.object_ref = object_ref
        self.attrs: Dict[str, "Attribute"] = {}

        self.return_type_name = return_type_name
        self.client = client

    def __call__(
        self,
        path: Union[List[str], str],
        index: int = 0,
        obj_type: Optional[type] = None,
    ) -> Optional[Union[Any, CallableT]]:
        raise NotImplementedError

    def _extract_attr_type(
        self,
        container: Union[
            List["ast.property.Property"],
            List["ast.klass.Class"],
            List["ast.module.Module"],
        ],
        field: str,
    ) -> None:
        for ref in self.attrs.values():
            sub_prop = getattr(ref, field, None)
            if sub_prop is None:
                continue

            container.extend(sub_prop)

    @property
    def classes(self) -> List["ast.klass.Class"]:
        out: List["ast.klass.Class"] = []

        if isinstance(self, ast.klass.Class):
            out.append(self)

        self._extract_attr_type(out, "classes")
        return out

    @property
    def properties(self) -> List["ast.property.Property"]:
        out: List["ast.property.Property"] = []

        if isinstance(self, ast.property.Property):
            out.append(self)

        self._extract_attr_type(out, "properties")
        return out

    def query(
        self, path: Union[List[str], str], obj_type: Optional[type] = None
    ) -> "Attribute":
        if obj_type is not None:
            if obj_type in self.lookup_cache:
                path = self.lookup_cache[obj_type]

        _path: List[str] = path if isinstance(path, list) else path.split(".")

        if len(_path) == 0:
            return self

        if _path[0] in self.attrs:
            return self.attrs[_path[0]].query(path=_path[1:])

        raise ValueError(f"Path {'.'.join(_path)} not present in the AST.")

    @property
    def name(self) -> str:
        path_and_name = self.path_and_name if self.path_and_name else ""
        return path_and_name.rsplit(".", maxsplit=1)[-1]

    def add_path(
        self,
        path: List[str],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[ModuleType] = None,
        is_static: bool = False,
    ) -> None:
        raise NotImplementedError
