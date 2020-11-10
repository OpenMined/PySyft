# stdlib
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Union

# syft relative
from .callable import Callable
from .module import Module
from .util import unsplit


class Globals(Module):

    _copy: Optional["copyType"]
    """The collection of frameworks held in a global namespace"""

    def __init__(self) -> None:
        super().__init__("globals")

    def __call__(
        self,
        path: Union[str, List[str]] = [],
        index: int = 0,
        return_callable: bool = False,
        obj_type: Optional[type] = None,
    ) -> Optional[Union[Callable, CallableT]]:
        if isinstance(path, str):
            path = path.split(".")
        return self.attrs[path[index]](
            path=path,
            index=index + 1,
            return_callable=return_callable,
            obj_type=obj_type,
        )

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int = 0,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[Union[Callable, CallableT]] = None,
    ) -> None:
        if isinstance(path, str):
            path = path.split(".")

        framework_name = path[index]

        if framework_name not in self.attrs:
            if framework_reference is not None:
                self.attrs[framework_name] = Module(
                    name=framework_name,
                    path_and_name=unsplit(path),
                    ref=framework_reference,
                    return_type_name=return_type_name,
                )
            else:
                raise Exception(
                    "You must pass in a framework object the first time you add method "
                    "within a framework."
                )

        attr = self.attrs[framework_name]
        if hasattr(attr, "add_path"):
            attr.add_path(  # type: ignore
                path=path, index=1, return_type_name=return_type_name
            )

    def copy(self) -> Optional["Globals"]:
        if self._copy is not None:
            return self._copy()
        return None


copyType = CallableT[[], Globals]
