"""This module contains Attribute, an interface of a generic node in the AST."""

# stdlib
from types import ModuleType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# relative
from .. import ast
from ..core.node.abstract.node import AbstractNodeClient
from ..logger import traceback_and_raise


class Attribute:
    """Attribute is the interface of a generic node in the AST that covers basic functionality."""

    __slots__ = [
        "path_and_name",
        "object_ref",
        "attrs",
        "return_type_name",
        "client",
        "_parent",
    ]

    lookup_cache: Dict[Any, Any] = {}

    def __init__(
        self,
        client: Optional[AbstractNodeClient],
        path_and_name: Optional[str] = None,
        object_ref: Any = None,
        return_type_name: Optional[str] = None,
        parent: Optional["Attribute"] = None,
    ):
        """Base constructor for all AST nodes.

        Args:
            client: The client for which all computation is being executed.
            path_and_name: The path for the current node, e.g. `syft.lib.python.List`.
            object_ref: The actual python object for which the computation is being made.
            return_type_name: The given action's return type name, with its full path, in string format.
            parent: The parent node in the AST.
        """
        self.client: Optional[AbstractNodeClient] = client
        self.path_and_name: Optional[str] = path_and_name
        self.object_ref: Any = object_ref
        self.return_type_name: Optional[str] = return_type_name

        # The `attrs` are the nodes that have the current node as a parent node
        # maps from the name on the path ot the actual attribute.
        self.attrs: Dict[str, "Attribute"] = {}
        self._parent: Optional["Attribute"] = parent

    def __call__(
        self,
        path: Union[List[str], str],
        index: int = 0,
        obj_type: Optional[type] = None,
    ) -> Any:
        """Execute the given node object reference with the given parameters.

        Args:
            path: The node path in AST to execute, e.g. `syft.lib.python.List` or ["syft", "lib", "python", "List].
            index: The associated position in the path for the current node.
            obj_type: The type of the object to be called, whose path is resolved from the `lookup_cache`.

        Returns:
            The results of running the computation on the object ref.
        """
        traceback_and_raise(NotImplementedError)

    def _extract_attr_type(
        self,
        container: Union[
            List["ast.klass.Class"],
            List["ast.module.Module"],
            List["ast.property.Property"],
        ],
        field: str,
    ) -> None:
        """Helper function to extract a class of nodes whose parent is the current node.

        Args:
            container: A list of objects in which we want to store the results.
            field: The typeof attribute from the current node's `attrs`.
        """
        for ref in self.attrs.values():
            sub_prop = getattr(ref, field, None)
            if sub_prop is None:
                continue

            container.extend(sub_prop)

    @property
    def classes(self) -> List["ast.klass.Class"]:
        """Extract all classes from the current node attributes.

        Returns:
            The list of classes in the current AST node attributes.
        """
        out = []

        if isinstance(self, ast.klass.Class):
            out.append(self)

        self._extract_attr_type(out, "classes")
        return out

    @property
    def properties(self) -> List["ast.property.Property"]:
        """Extract all properties from the current node attributes.

        Returns:
            The list of properties in the current AST node attributes.
        """
        out = []

        if isinstance(self, ast.property.Property):
            out.append(self)

        self._extract_attr_type(out, "properties")
        return out

    def query(
        self, path: Union[List[str], str], obj_type: Optional[type] = None
    ) -> "Attribute":
        """The query method is a tree traversal function based on the path to retrieve the node.

           It has a similar functionality to `__call__`,
           main difference being that `query` retrieves node without performing execution on node.

        Args:
            path: The node path in AST to be queried, e.g. `syft.lib.python.List` or ["syft", "lib", "python", "List"].
            obj_type: The type of the object that we want to call, whose path is resolved from the `lookup_cache`.

        Returns:
            The attribute in the AST at the given initial path.
        """
        # TODO: fix hacky work around
        if path == "syft.lib.python.list.List":
            path = "syft.lib.python.List"

        if obj_type is not None:
            # If the searched given type has already been seen, resolve it with the path from `lookup_cache`.
            if obj_type in self.lookup_cache:
                path = self.lookup_cache[obj_type]

        _path = path if isinstance(path, list) else path.split(".")

        if len(_path) == 0:
            return self

        # If the first element of the path is a child node, continue the query in the child node
        if _path[0] in self.attrs:
            return self.attrs[_path[0]].query(path=_path[1:])

        traceback_and_raise(
            ValueError(f"Path {'.'.join(_path)} not present in the AST.")
        )

    @property
    def name(self) -> str:
        """Retrieve the name of the current AST node from its `path_and_name`.

        Returns:
            The name of the current attribute.
        """
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
        """The add_path method adds new nodes in AST based on type of current node and type of object to be added.

        Args:
            path: The node path added in AST, e.g. `syft.lib.python.List` or ["syft", "lib", "python", "List].
            index: The associated position in the path for the current node.
            framework_reference: The Python framework in which we can resolve same path to obtain Python object.
            return_type_name: The return type name of the given action as a string with its full path.
            is_static: If the queried object is static, it has to be found on AST itself, not on an existing pointer.
        """
        traceback_and_raise(NotImplementedError)

    def fetch_live_object(self) -> Any:
        """Get the new object and its attributes from the client."""
        return getattr(self.parent.object_ref, self.name)

    def object_change(self) -> bool:
        """Check if client wants to change any nodes in the AST with a new object."""
        return id(self.fetch_live_object()) != id(self.object_ref)

    def reconstruct_node(self) -> None:
        """Changes node reference in the AST by adding the new object's reference as specified by client."""
        self.object_ref = self.fetch_live_object()

    def apply_node_changes(self) -> None:
        """Apply the changes in the nodes in the AST as specified by the client."""
        if self._parent and self.object_change():
            self.reconstruct_node()

    @property
    def parent(self) -> "Attribute":
        """Check if all the nodes have a parent node.

        Returns:
            Attribute: parent node

        Raises:
            AttributeError: If node has no parent attribute.
        """
        if self._parent:
            return self._parent

        raise AttributeError(f"Node {self} in the AST has not parent attribute set!")
