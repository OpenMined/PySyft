# stdlib
from types import ModuleType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# syft relative
from .. import ast
from ..core.node.abstract.node import AbstractNodeClient


class Attribute:
    """
    Attribute is the interface of a generic node in the AST that covers basic functionality.
    """

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
        client: Optional[AbstractNodeClient],
        path_and_name: str,
        object_ref: Any = None,
        return_type_name: Optional[str] = None,
    ):
        """
        Base constructor for all AST nodes.

         Args:
             client (Optional[AbstractNodeClient]): The client for which all computation is being executed.
             path_and_name (str): The path for the current node. Eg. `syft.lib.python.List`
             object_ref (Any): The actual python object for which the computation is being made.
             return_type_name (Optional[str]): The return type name of the given action as a
                 string (the full path to it, similar to path_and_name).
        """
        self.client: Optional[AbstractNodeClient] = client
        self.path_and_name: Optional[str] = path_and_name
        self.object_ref: Any = object_ref
        self.return_type_name: Optional[str] = return_type_name

        # the attrs attribute are the nodes that have the current node as a parent node
        # maps from the name on the path ot the actual attribute.
        self.attrs: Dict[str, "Attribute"] = {}

    def __call__(
        self,
        path: Union[List[str], str],
        index: int = 0,
        obj_type: Optional[type] = None,
    ) -> Any:
        """
        The __call__ method executes the given node object reference with the given parameters.

         Args:
             path (Union[List[str], str]): The path for the node in the AST to be executed. Eg.
                 `syft.lib.python.List` or ["syft", "lib", "python", "List]
             index (int): The associated position in the path for the current node.
             obj_type (Optional[type]): The type of the object that we want to call,
                 solving directly the path from the lookup_cache.

         Returns:
             Any: The results of running the computation on the object ref.
        """
        raise NotImplementedError

    def _extract_attr_type(
        self,
        container: Union[
            List["Attribute"],
        ],
        field: str,
    ) -> None:
        """
        Helper function to extract a class of nodes out of the current node.

         Args:
             container (List[Attribute]): A list of objects in which we want to store the
                 results.
             field (str): The typeof attribute from the current node attrs.

         Returns:
             Any: The results of running the computation on the object ref.
        """

        for ref in self.attrs.values():
            sub_prop = getattr(ref, field, None)
            if sub_prop is None:
                continue

            for sub in sub_prop:
                container.append(sub)

    @property
    def classes(self) -> List["ast.klass.Class"]:
        """
        Property to extract all classes from the current node attributes.

        Returns:
            List["ast.klass.Class"]: the list of classes in the current AST node attributes.
        """
        out: List["ast.klass.Class"] = []

        if isinstance(self, ast.klass.Class):
            out.append(self)

        self._extract_attr_type(out, "classes")
        return out

    @property
    def properties(self) -> List["ast.property.Property"]:
        """
        Property to extract all properties from the current node attributes.

        Returns:
            List["ast.klass.Property"]: the list of properties in the current AST node attributes.
        """
        out: List["ast.property.Property"] = []

        if isinstance(self, ast.property.Property):
            out.append(self)

        self._extract_attr_type(out, "properties")
        return out

    def query(
        self, path: Union[List[str], str], obj_type: Optional[type] = None
    ) -> "Attribute":
        """
        The query method is a tree traversal function based on the path to retrieve the node. It
        has a similar functionality to __call__, the main difference being that this retrieves the
        node without any execution on it.

         Args:
              path (Union[List[str], str]): The path for the node in the AST to be queried. Eg.
              `syft.lib.python.List` or ["syft", "lib", "python", "List]

              obj_type (Optional[type]): The type of the object that we want to call,
              solving directly the path from the lookup_cache.

         Returns:
             Attribute: The attribute in the AST at the given initial path.
        """

        if obj_type is not None:
            # if the searched given type has already been seen, solve it a known path.
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
        """
        The name property retrieves the name of the current AST node from the path_and_name.

         Returns:
             str: The name of the current attribute.
        """

        return self.path_and_name.rsplit(".", maxsplit=1)[-1]

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[ModuleType] = None,
        is_static: bool = False,
    ) -> None:
        """
        The add_path method adds new nodes in the AST based on the type of the current node and
        the type of the object to be added.

         Args:
              path (Union[List[str], str]): The path for the node in the AST to be added. Eg.
                  `syft.lib.python.List` or ["syft", "lib", "python", "List]
               index (int): The associated position in the path for the current node.
              framework_reference(Optional[ModuleType]):The python framework in which we can solve
                   the same path to obtain the python object.
              return_type_name (Optional[str]): The return type name of the given action as a
                 string (the full path to it, similar to path_and_name).
              is_static (bool): if the queried object is static, it has to be found on the ast
                itself, not on an existing pointer.
        """
        raise NotImplementedError
