# future
from __future__ import annotations

# stdlib
import importlib
import inspect
from inspect import Signature
from inspect import _signature_fromstr
from types import BuiltinFunctionType
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

# third party
import jax
from jaxlib.xla_extension import CompiledFunction
import numpy
from typing_extensions import Self

# relative
from .lib_permissions import ALL_EXECUTE
from .lib_permissions import CMPPermission
from .lib_permissions import NONE_EXECUTE
from .signature import get_signature
from .serializable import serializable

LIB_IGNORE_ATTRIBUTES = set(
    ["os", "__abstractmethods__", "__base__", " __bases__", "__class__"]
)


def import_from_path(path: str) -> type:
    if "." in path:
        top_level_module, attr_path = path.split(".", 1)
    else:
        top_level_module = path
        attr_path = ""

    res = importlib.import_module(top_level_module)
    path_parts = [x for x in attr_path.split(".") if x != ""]
    for attr in path_parts:
        res = getattr(res, attr)
    return res  # type: ignore

@serializable()
class CMPBase:
    """cmp: cascading module permissions"""

    def __init__(
        self,
        path: str,
        children: Optional[Union[List, Dict]] = None,
        permissions: Optional[CMPPermission] = None,
        obj: Optional[Any] = None,
        absolute_path: Optional[str] = None,
        text_signature: Optional[str] = None,
    ):
        self.permissions: Optional[CMPPermission] = permissions
        self.path: str = path
        self.obj: Optional[Any] = obj if obj is not None else None
        self.absolute_path = absolute_path
        self.signature: Optional[Signature] = None

        self.children: Dict[str, CMPBase] = dict()
        if isinstance(children, list):
            self.children = {f"{c.path}": c for c in children}
        elif isinstance(children, dict):
            self.children = children

        for c in self.children.values():
            if c.absolute_path is None:
                c.absolute_path = f"{path}.{c.path}"

        if text_signature is not None:
            self.signature = _signature_fromstr(
                inspect.Signature, obj, text_signature, True
            )

        self.is_built = False

    def set_signature(self) -> None:
        pass

    def build(self, stack: List, root: Optional[CMPBase] = None) -> None:
        if self.obj is None:
            self.obj = import_from_path(self.absolute_path)

        if root is None:
            root = self

        if self.signature is None:
            self.set_signature()

        child_paths = set([p for p in self.children.keys()])

        for attr_name in getattr(self.obj, "__dict__", dict()).keys():
            if attr_name not in LIB_IGNORE_ATTRIBUTES:
                if attr_name in child_paths:
                    child = self.children[attr_name]
                else:
                    try:
                        attr = getattr(self.obj, attr_name)
                    except Exception:  # nosec
                        continue
                    child = self.init_child(  # type: ignore
                        root.obj,
                        f"{self.path}.{attr_name}",
                        attr,
                        f"{self.absolute_path}.{attr_name}",
                    )
                if child is not None:
                    stack.append(child)
                    self.children[attr_name] = child

    def __getattr__(self, __name: str) -> Any:
        if __name in self.children:
            return self.children[__name]
        else:
            raise ValueError(f"property {__name} not defined")

    def init_child(
        self,
        parent_obj: Union[type, object],
        child_path: str,
        child_obj: Union[type, object],
        absolute_path: str,
    ) -> Optional[CMPBase]:
        """Get the child of parent as a CMPBase object

        Args:
            parent_obj (_type_): parent object
            child_path (_type_): _description_
            child_obj (_type_): _description_

        Returns:
            _type_: _description_
        """  # If the child is not a module, then
        is_child_valid = CMPBase.check_package_membership(parent_obj, child_obj)
        if not is_child_valid:
            return None

        if inspect.ismodule(child_obj):
            ## TODO, we could register modules and functions in 2 ways:
            # A) as numpy.float32 (what we are doing now)
            # B) as numpy.core.float32 (currently not supported)
            # only allow submodules
            return CMPModule(
                child_path,
                permissions=self.permissions,
                obj=child_obj,
                absolute_path=absolute_path,
            )  # type: ignore

        # Here we have our own isfunction as there are multiple callable objects worth considering
        if CMPBase.isfunction(child_obj):
            return CMPFunction(
                child_path,
                permissions=self.permissions,
                obj=child_obj,
                absolute_path=absolute_path,
            )  # type: ignore

        if inspect.isclass(child_obj):
            return CMPClass(
                child_path,
                permissions=self.permissions,
                obj=child_obj,
                absolute_path=absolute_path,
            )  # type: ignore

        # default case if we didnt cover it
        # currently used for objects
        return CMPBase(
            child_path,
            permissions=self.permissions,
            obj=child_obj,
            absolute_path=absolute_path,
        )

    @staticmethod
    def check_package_membership(parent_obj: Any, child_obj: Any) -> bool:
        # we are wrapping this as some objects might not have some of the dunder methods
        # TODO: check if that is truly necessary
        try:
            # if we find the same name we should avoid a circular import (probably obsolete)
            if child_obj.__name__ == parent_obj.__name__:
                return False

            # if the name of the parent can be found at the start of the name of the child we are good
            if child_obj.__name__.startswith(parent_obj.__name__):
                return True

            # if the child has a module, then we should just make sure it has the same start
            # as the parent, but not the entire name, as there might be relative imports
            # in other parts of the codebase
            if hasattr(child_obj, "__module__"):
                return child_obj.__module__.startswith(
                    parent_obj.__name__.split(".")[0]
                )
            else:
                # same idea as the with the child name
                # TODO: this is a fix for for instance numpy ufuncs
                return child_obj.__class__.__module__.startswith(parent_obj.__name__)
        except Exception:  # nosec
            pass
        return False

    def flatten(self) -> List[Self]:
        res = [self]
        for c in self.children.values():
            res += c.flatten()
        return res

    @staticmethod
    def isfunction(obj: Callable) -> bool:
        return (
            inspect.isfunction(obj)
            or type(obj) == numpy.ufunc
            or isinstance(obj, BuiltinFunctionType)
            or isinstance(obj, CompiledFunction)
        )

    def __repr__(
        self, indent: int = 0, is_last: bool = False, parent_path: str = ""
    ) -> str:
        """Visualize the tree, e.g.:
        ├───numpy (ALL_EXECUTE)
        │    ├───ModuleDeprecationWarning (ALL_EXECUTE)
        │    ├───VisibleDeprecationWarning (ALL_EXECUTE)
        │    ├───_CopyMode (ALL_EXECUTE)
        │    ├───compat (ALL_EXECUTE)
        │    ├───core (ALL_EXECUTE)
        │    │    ├───_ufunc_reconstruct (ALL_EXECUTE)
        │    │    ├───_DType_reconstruct (ALL_EXECUTE)
        │    │    └───__getattr__ (ALL_EXECUTE)
        │    ├───char (ALL_EXECUTE)
        │    │    ├───_use_unicode (ALL_EXECUTE)
        │    │    ├───_to_string_or_unicode_array (ALL_EXECUTE)
        │    │    ├───_clean_args (ALL_EXECUTE)

        Args:
            indent (int, optional): indentation level. Defaults to 0.
            is_last (bool, optional): is last item of collection. Defaults to False.
            parent_path (str, optional): path of the parent obj. Defaults to "".

        Returns:
            str: representation of the CMP
        """
        last_idx, c_indent = len(self.children) - 1, indent + 1
        children_string = "".join(
            [
                c.__repr__(c_indent, is_last=i == last_idx, parent_path=self.path)  # type: ignore
                for i, c in enumerate(
                    sorted(
                        self.children.values(),
                        key=lambda x: x.permissions.permission_string,  # type: ignore
                    )  # type: ignore
                )  # type: ignore
            ]
        )
        tree_prefix = "└───" if is_last else "├───"
        indent_str = "│    " * indent + tree_prefix
        if parent_path != "":
            path = self.path.replace(f"{parent_path}.", "")
        else:
            path = self.path
        return f"{indent_str}{path} ({self.permissions})\n{children_string}"

    def get_path(self, path: str) -> Any:
        segments = path.split(".")
        root = segments[0]
        if root in self.children:
            if len(segments) == 1:
                return self.children[root]
            else:
                return self.children[root].get_path(".".join(segments[1:]))
        else:
            raise ValueError(f"property {path} does not exist")

    def rebuild(self):
        self.obj = import_from_path(self.absolute_path)
        for child in self.children.values():
            child.rebuild()
        
    def reset_objs(self):
        self.obj = None
        for child in self.children.values():
            child.reset_objs()
        


@serializable()
class CMPModule(CMPBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

@serializable()
class CMPFunction(CMPBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_signature()

    @property
    def name(self) -> str:
        return self.obj.__name__  # type: ignore

    def set_signature(self) -> None:
        try:
            self.signature = get_signature(self.obj)
        except Exception:  # nosec
            pass

@serializable()
class CMPClass(CMPBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_signature()

    @property
    def name(self) -> str:
        # possibly change to
        # func_name = path.split(".")[-1]
        return self.obj.__name__  # type: ignore

    def set_signature(self) -> None:
        try:
            self.signature = get_signature(self.obj)
        except Exception:  # nosec
            try:
                self.signature = get_signature(self.obj.__init__)  # type: ignore
            except Exception:  # nosec
                pass

@serializable()
class CMPMethod(CMPBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

@serializable()
class CMPProperty(CMPBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

@serializable()
class CMPTree:
    """root node of the Tree(s), with one child per library"""

    def __init__(self, children: List[CMPModule]):
        self.children = {c.path: c for c in children}

    def build(self) -> Self:
        self.stack = []
        for c in self.children.values():
            c.absolute_path = c.path
            self.stack.append(c)
        while len(self.stack) > 0:
            c = self.stack.pop(0)
            c.build(stack=self.stack, root=c)
        return self

    def flatten(self) -> Sequence[CMPBase]:
        res = []
        for c in self.children.values():
            res += c.flatten()
        return res

    def __getattr__(self, _name: str) -> Any:
        if _name in self.children:
            return self.children[_name]
        else:
            raise ValueError(f"property {_name} does not exist")

    def __repr__(self) -> str:
        return "\n".join([c.__repr__() for c in self.children.values()])

    def get_path(self, path: str) -> Any:
        segments = path.split(".")
        root = segments[0]
        if root in self.children:
            if len(segments) == 1:
                return self.children[root]
            else:
                return self.children[root].get_path(".".join(segments[1:]))
        else:
            raise ValueError(f"property {path} does not exist")

# @serializable()
# class PathWrapper(syftObject)


action_execute_registry_libs = CMPTree(
    children=[
        # CMPModule(
        #     "numpy",
        #     permissions=ALL_EXECUTE,
        #     children=[
        #         CMPFunction(
        #             "concatenate",
        #             permissions=ALL_EXECUTE,
        #             text_signature="concatenate(a1,a2, *args,axis=0,out=None,dtype=None,casting='same_kind')",
        #         ),
        #         CMPFunction("source", permissions=NONE_EXECUTE),
        #         CMPFunction("fromfile", permissions=NONE_EXECUTE),
        #         CMPFunction(
        #             "set_numeric_ops",
        #             permissions=ALL_EXECUTE,
        #             text_signature="set_numeric_ops(op1,op2, *args)",
        #         ),
        #         CMPModule("testing", permissions=NONE_EXECUTE),
        #     ],
        # ),
        # CMPModule(
        #     "jax",
        #     permissions=ALL_EXECUTE,
        # ),
    ]
).build()