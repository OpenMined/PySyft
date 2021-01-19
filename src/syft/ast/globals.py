# stdlib
from types import ModuleType
from typing import Any
from typing import Callable as CallableT
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# syft relative
from ..core.common.uid import UID
from .callable import Callable
from .module import Module


class Globals(Module):
    _copy: Optional["copyType"]
    registered_clients: Dict[UID, Any] = {}
    loaded_lib_constructors: Dict[str, CallableT] = {}

    """The collection of frameworks held in the global namespace"""

    def __call__(
        self,
        path: Union[List[str], str],
        index: int = 0,
        obj_type: Optional[type] = None,
    ) -> Optional[Union[Callable, CallableT]]:

        _path: List[str] = (
            path.split(".") if isinstance(path, str) else path if path else []
        )

        if not _path:
            raise ValueError("NAKE PROPER SCHEMA")

        return self.attrs[_path[index]](path=_path, index=index + 1, obj_type=obj_type)

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int = 0,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[ModuleType] = None,
        is_static: bool = False,
    ) -> None:
        if isinstance(path, str):
            path = path.split(".")

        framework_name = path[index]

        if framework_name not in self.attrs:
            if framework_reference is None:
                raise Exception(
                    "You must pass in a framework object, the first time you add method \
                    within the framework."
                )

            self.attrs[framework_name] = Module(
                path_and_name=".".join(path[: index + 1]),
                object_ref=framework_reference,
                return_type_name=return_type_name,
                client=self.client,
            )

        attr = self.attrs[framework_name]
        attr.add_path(  # type: ignore
            path=path, index=1, return_type_name=return_type_name
        )

    def copy(self, client) -> Optional["Globals"]:
        if self._copy is not None:
            return self._copy(client)
        return None

    def register_updates(self, client: Any) -> None:
        # any previously loaded libs need to be applied
        for _, update_ast in self.loaded_lib_constructors.items():
            update_ast(ast=client)

        # make sure to get any future updates
        self.registered_clients[client.id] = client


copyType = CallableT[[], Globals]
