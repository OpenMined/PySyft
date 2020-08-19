# from .. import ast # CAUSES Circular import errors
from typing import Union, List, Optional
from .module import Module
from .util import unsplit


class Globals(Module):

    """The collection of frameworks held in a global namespace"""

    def __init__(self) -> None:
        super().__init__("globals", None, None, None)

    def add_framework(self, attr_name: str, attr: object) -> None:
        self.attrs[attr_name] = attr

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int = 0,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[object] = None,
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

        self.attrs[framework_name].add_path(
            path=path, index=1, return_type_name=return_type_name
        )
