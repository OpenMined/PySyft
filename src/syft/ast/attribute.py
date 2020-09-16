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
        is_property: bool = False,
    ):
        self.name = name  # __add__
        self.path_and_name = path_and_name  # torch.Tensor.__add__
        self.ref = ref  # <the actual add method object>
        self.attrs: Dict[
            str, Union[ast.callable.Callable, CallableT]
        ] = {}  # any attrs of __add__ ... is none in this case
        self.return_type_name = return_type_name
        self.is_property = is_property

    def set_client(self, client: Any) -> None:
        self.client = client
        for _, attr in self.attrs.items():
            if hasattr(attr, "set_client"):
                attr.set_client(client=client)  # type: ignore

    @property
    def classes(self) -> List["ast.klass.Class"]:
        out: List[ast.klass.Class] = list()

        if isinstance(self, ast.klass.Class):
            out.append(self)

        for _, ref in self.attrs.items():
            sub_prop = getattr(ref, "classes", None)
            if sub_prop is not None:
                for sub in sub_prop:
                    out.append(sub)
        return out

    @property
    def methods(self) -> List["ast.method.Method"]:
        out: List[ast.method.Method] = []

        if isinstance(self, ast.method.Method):
            out.append(self)

        for _, ref in self.attrs.items():
            sub_prop = getattr(ref, "methods", None)
            if sub_prop is not None:
                for sub in sub_prop:
                    out.append(sub)
        return out

    @property
    def functions(self) -> List["ast.function.Function"]:
        out: List[ast.function.Function] = []

        if isinstance(self, ast.function.Function):
            out.append(self)

        for _, ref in self.attrs.items():
            sub_prop = getattr(ref, "functions", None)
            if sub_prop is not None:
                for sub in sub_prop:
                    out.append(sub)
        return out

    @property
    def modules(self) -> List["ast.module.Module"]:
        out: List[ast.module.Module] = []

        if isinstance(self, ast.module.Module):
            out.append(self)

        for _, ref in self.attrs.items():
            sub_prop = getattr(ref, "modules", None)
            if sub_prop is not None:
                for sub in sub_prop:
                    out.append(sub)
        return out
