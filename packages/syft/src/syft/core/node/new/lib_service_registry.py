# stdlib
import importlib
import inspect
from inspect import Signature
from types import BuiltinFunctionType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
import numpy
from typing_extensions import Self

# relative
from .lib_permissions import ALL_EXECUTE
from .lib_permissions import CMPPermission
from .lib_permissions import NONE_EXECUTE
from .signature import get_signature

LIB_IGNORE_ATTRIBUTES = set(
    ["os", "__abstractmethods__", "__base__", " __bases__", "__class__"]
)


def import_from_path(path):
    if "." in path:
        top_level_module, attr_path = path.split(".", 1)
    else:
        top_level_module = path
        attr_path = ""

    res = importlib.import_module(top_level_module)
    path_parts = [x for x in attr_path.split(".") if x != ""]
    for attr in path_parts:
        res = getattr(res, attr)
    return res


class CMPBase:
    """cmp: cascading module permissions"""

    def __init__(
        self,
        path,
        children=None,
        permissions: Optional[CMPPermission] = None,
        manual_signature: Optional[Signature] = None,
        obj: Optional[Any] = None,
        absolute_path=None,
    ):
        self.permissions: Optional[CMPPermission] = permissions
        self.path: str = path
        self.obj: Optional[Any] = obj if obj is not None else None
        self.absolute_path = absolute_path

        self.children: Dict[str, CMPBase] = dict()
        if isinstance(children, list):
            self.children = {f"{c.path}": c for c in children}
        elif isinstance(children, dict):
            self.children = children

        for c in self.children.values():
            if c.absolute_path is None:
                c.absolute_path = f"{path}.{c.path}"

        if manual_signature is not None:
            self.signature: Signature = manual_signature
        else:
            self.signature = None

        self.is_built = False

    def set_signature(self):
        pass

    def build(self):
        if self.obj is None:
            self.obj = import_from_path(self.absolute_path)

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
                    except Exception:
                        continue
                    child = self.init_child(
                        self.obj,
                        f"{self.path}.{attr_name}",
                        attr,
                        f"{self.absolute_path}.{attr_name}",
                    )
                if child is not None:
                    child.build()
                    self.children[attr_name] = child

    def __getattr__(self, __name: str) -> Any:
        if __name in self.children:
            return self.children[__name]
        else:
            raise ValueError(f"property {__name} not defined")

    def init_child(self, parent_obj, child_path: str, child_obj, absolute_path):
        """Get the child of parent as a CMPBase object

        Args:
            parent_obj (_type_): parent object
            child_path (_type_): _description_
            child_obj (_type_): _description_

        Returns:
            _type_: _description_
        """
        parent_is_parent_module = CMPBase.parent_is_parent_module(parent_obj, child_obj)
        if CMPBase.isfunction(child_obj) and parent_is_parent_module:
            return CMPFunction(
                child_path,
                permissions=self.permissions,
                obj=child_obj,
                absolute_path=absolute_path,
            )
        elif inspect.ismodule(child_obj) and CMPBase.is_submodule(
            parent_obj, child_obj
        ):
            ## TODO, we could register modules and functions in 2 ways:
            # A) as numpy.float32 (what we are doing now)
            # B) as numpy.core.float32 (currently not supported)
            # only allow submodules

            return CMPModule(
                child_path,
                permissions=self.permissions,
                obj=child_obj,
                absolute_path=absolute_path,
            )
            # register_lib_func(path, lib_obj)
        elif inspect.isclass(child_obj) and parent_is_parent_module:
            return CMPClass(
                child_path,
                permissions=self.permissions,
                obj=child_obj,
                absolute_path=absolute_path,
            )
            # register_lib_class(path, lib_obj)
        else:
            return None

    @staticmethod
    def is_submodule(parent, child):
        try:
            if "." not in child.__package__:
                return False
            else:
                child_parent_module = child.__package__.rsplit(".", 1)[0]
                if parent.__package__ == child_parent_module:
                    return True
                else:
                    return False
        except Exception:
            return False

    @staticmethod
    def parent_is_parent_module(parent_obj, child_obj):
        try:
            if hasattr(child_obj, "__module__"):
                return child_obj.__module__ == parent_obj.__name__
            else:
                # TODO: this is a fix for for instance numpy ufuncs
                return child_obj.__class__.__module__ == parent_obj.__name__
        except Exception:
            return False

    def flatten(self) -> List[Self]:
        res = [self]
        for c in self.children.values():
            res += c.flatten()
        return res

    @staticmethod
    def isfunction(obj):
        return (
            inspect.isfunction(obj)
            or type(obj) == numpy.ufunc
            or isinstance(obj, BuiltinFunctionType)
        )

    def __repr__(self, indent=0, is_last=False, parent_path=""):
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
                c.__repr__(c_indent, is_last=i == last_idx, parent_path=self.path)
                for i, c in enumerate(
                    sorted(
                        self.children.values(),
                        key=lambda x: x.permissions.permission_string,
                    )
                )
            ]
        )
        tree_prefix = "└───" if is_last else "├───"
        indent_str = "│    " * indent + tree_prefix
        if parent_path != "":
            path = self.path.replace(f"{parent_path}.", "")
        else:
            path = self.path
        return f"{indent_str}{path} ({self.permissions})\n{children_string}"


class CMPModule(CMPBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CMPFunction(CMPBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_signature()

    @property
    def name(self):
        return self.obj.__name__

    def set_signature(self):
        try:
            self.signature = get_signature(self.obj)
        except Exception:
            pass


class CMPClass(CMPBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_signature()

    @property
    def name(self):
        # possibly change to
        # func_name = path.split(".")[-1]
        return self.obj.__name__

    def set_signature(self):
        try:
            self.signature = get_signature(self.obj)
        except Exception:
            try:
                self.signature = get_signature(self.obj.__init__)
            except Exception:
                pass


class CMPMethod(CMPBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CMPProperty(CMPBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CMPTree:
    """root node of the Tree(s), with one child per library"""

    def __init__(self, children: List[CMPModule]):
        self.children = {c.path: c for c in children}

    def build(self):
        for c in self.children.values():
            c.absolute_path = c.path
            c.build()
        return self

    def flatten(self) -> List[CMPBase]:
        res = []
        for c in self.children.values():
            res += c.flatten()
        return res

    def __getattr__(self, _name):
        if _name in self.children:
            return self.children[_name]
        else:
            raise ValueError(f"property {_name} does not exist")

    def __repr__(self):
        return "\n".join([c.__repr__() for c in self.children.values()])


action_execute_registry_libs = CMPTree(
    children=[
        CMPModule(
            "numpy",
            permissions=ALL_EXECUTE,
            children=[
                CMPModule("testing", permissions=NONE_EXECUTE),
            ],
        ),
    ]
).build()
