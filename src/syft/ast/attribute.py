# stdlib
from abc import ABC
from typing import Any
from typing import Callable as CallableT
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# syft relative
from .. import ast


class Attribute(ABC):
    client: Optional[Any]

    def __init__(
        self,
        name: Optional[str] = None,
        path_and_name: Optional[str] = None,
        ref: Optional[Union["ast.callable.Callable", CallableT]] = None,
        return_type_name: Optional[str] = None,
    ):
        self.name = name
        self.path_and_name = path_and_name
        self.ref = ref
        self.attrs: Dict[str, Union[ast.callable.Callable, CallableT]] = {}
        self.return_type_name = return_type_name

    def set_client(self, client: Any) -> None:
        self.client = client
        for attr in self.attrs.values():
            if hasattr(attr, "set_client"):
                attr.set_client(client=client)  # type: ignore

    def _extract_attr_type(
        self,
        container: Union[
            List["ast.klass.Class"],
            List["ast.module.Module"],
        ],
        field: str,
    ) -> None:
        for ref in self.attrs.values():
            sub_prop = getattr(ref, field, None)
            if sub_prop is None:
                continue

            for sub in sub_prop:
                container.append(sub)

    @property
    def classes(self) -> List["ast.klass.Class"]:
        out: List["ast.klass.Class"] = []

        if isinstance(self, ast.klass.Class):
            out.append(self)

        self._extract_attr_type(out, "classes")
        return out

    @property
    def modules(self) -> List["ast.module.Module"]:
        out: List["ast.module.Module"] = []

        if isinstance(self, ast.module.Module):
            out.append(self)

        self._extract_attr_type(out, "modules")
        return out

    def query(self, path: List[str]):
        if path == "":
            print("CE")

        if isinstance(path, str):
            path = path.split(".")

        if len(path) == 0:
            return self
        next = path[0]
        if next not in self.attrs:
            raise ValueError(f"Path {'.'.join(path)} not present in the AST.")

        return self.attrs[next].query(path[1:])
