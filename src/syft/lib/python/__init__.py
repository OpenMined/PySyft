# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import torch

# syft relative
from ...ast.globals import Globals
from ...ast.klass import Class
from ...ast.module import Module
from .bool import Bool
from .complex import Complex
from .dict import Dict
from .dict import DictWrapper
from .float import Float
from .int import Int
from .list import List
from .namedtuple import ValuesIndices
from .namedtuple import ValuesIndicesWrapper
from .none import SyNone
from .none import _SyNone
from .primitive_container import Any
from .primitive_interface import PyPrimitive
from .string import String
from .tuple import Tuple

for syft_type in [
    Bool,
    Complex,
    Dict,
    DictWrapper,
    Float,
    Int,
    SyNone,
    _SyNone,
    Any,
    PyPrimitive,
    String,
    Tuple,
    ValuesIndices,
    ValuesIndicesWrapper,
]:
    syft_type.__module__ = __name__


def get_parent(path: str, root: TypeAny) -> Module:
    parent = root
    for step in path.split(".")[:-1]:
        parent = parent.attrs[step]
    return parent


def add_modules(ast: Globals, modules: TypeList[str]) -> None:
    for module in modules:
        parent = get_parent(module, ast)
        attr_name = module.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Module(
                attr_name,
                module,
                None,
                return_type_name="",
            ),
        )


def add_classes(ast: Globals, paths: TypeList[TypeTuple[str, str, Any]]) -> None:
    for path, return_type, ref in paths:
        parent = get_parent(path, ast)
        attr_name = path.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Class(
                attr_name,
                path,
                ref,  # type: ignore
                return_type_name=return_type,
            ),
        )


def add_methods(ast: Globals, paths: TypeList[TypeTuple[str, str, Any]]) -> None:
    for path, return_type, _ in paths:
        parent = get_parent(path, ast)
        path_list = path.split(".")
        parent.add_path(
            path=path_list, index=len(path_list) - 1, return_type_name=return_type
        )


def create_python_ast() -> Globals:
    ast = Globals()

    modules = [
        "syft",
        "syft.lib",
        "syft.lib.python",
    ]
    classes = [
        ("syft.lib.python.Bool", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Complex", "syft.lib.python.Complex", Complex),
        ("syft.lib.python.Dict", "syft.lib.python.Dict", Dict),
        ("syft.lib.python.Float", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int", "syft.lib.python.Int", Int),
        ("syft.lib.python.List", "syft.lib.python.List", List),
        ("syft.lib.python.String", "syft.lib.python.String", String),
        ("syft.lib.python._SyNone", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.PyPrimitive", "syft.lib.python.PyPrimitive", PyPrimitive),
        ("syft.lib.python.Any", "syft.lib.python.Any", Any),
        ("syft.lib.python.Tuple", "syft.lib.python.Tuple", Tuple),
        (
            "syft.lib.python.ValuesIndices",
            "syft.lib.python.ValuesIndices",
            ValuesIndices,
        ),
    ]

    methods = [
        # List methods - quite there
        ("syft.lib.python.List.__len__", "syft.lib.python.Int", List),
        ("syft.lib.python.List.__getitem__", "syft.lib.python.Any", Any),
        ("syft.lib.python.List.__iter__", "syft.lib.python.Any", Any),
        ("syft.lib.python.List.__add__", "syft.lib.python.List", List),
        ("syft.lib.python.List.append", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.List.__gt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__lt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__le__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__ge__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__iadd__", "syft.lib.python.List", List),
        ("syft.lib.python.List.__imul__", "syft.lib.python.List", List),
        ("syft.lib.python.List.__iadd__", "syft.lib.python.List", List),
        ("syft.lib.python.List.__contains__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__delattr__", "syft.lib.python.None", _SyNone),
        ("syft.lib.python.List.__delitem__", "syft.lib.python.None", _SyNone),
        ("syft.lib.python.List.__eq__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__mul__", "syft.lib.python.List", List),
        ("syft.lib.python.List.__ne__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__sizeof__", "syft.lib.python.Int", Int),
        ("syft.lib.python.List.__len__", "syft.lib.python.Int", Int),
        ("syft.lib.python.List.__getitem__", "syft.lib.python.Any", Any),
        ("syft.lib.python.List.__setitem__", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.List.__rmul__", "syft.lib.python.List", List),
        ("syft.lib.python.List.copy", "syft.lib.python.List", List),
        ("syft.lib.python.List.count", "syft.lib.python.Int", Int),
        ("syft.lib.python.List.sort", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.List.reverse", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.List.remove", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.List.pop", "syft.lib.python.Any", Any),
        ("syft.lib.python.List.index", "syft.lib.python.Any", Any),
        ("syft.lib.python.List.insert", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.List.clear", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.List.extend", "syft.lib.python._SyNone", _SyNone),
        ("syft.lib.python.List.__reversed__", "syft.lib.python.Any", Any),
        ("syft.lib.python.List.__delitem__", "syft.lib.python._SyNone", _SyNone),
        # Bool methods - quite there
        ("syft.lib.python.Bool.__abs__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__eq__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__add__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__and__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__ceil__", "syft.lib.python.Bool", Int),
        ("syft.lib.python.Bool.__divmod__", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Bool.__floor__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__float__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Bool.__floordiv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__ge__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__gt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__invert__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__le__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__lt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__lshift__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__mod__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__mul__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__ne__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__neg__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__or__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__pos__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__pow__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__radd__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rand__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__rdivmod__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__rfloordiv__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__rlshift__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rmod__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rmul__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__ror__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Bool.__round__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rpow__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rrshift__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rshift__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rsub__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rtruediv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rxor__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__sub__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__truediv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__xor__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__trunc__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.conjugate", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.bit_length", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.as_integer_ratio", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Bool.numerator", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.real", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.imag", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.denominator", "syft.lib.python.Int", Int),
        # Float methods - subject to further change due
        ("syft.lib.python.Float.__add__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__truediv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__divmod__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__eq__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Float.__ge__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Float.__lt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Float.__le__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Float.__gt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Float.__add__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__abs__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__bool__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Float.__sub__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__rsub__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__mul__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__rmul__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__divmod__", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Float.__int__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Float.__neg__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__ne__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Float.__floordiv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__truediv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__mod__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__rmod__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__rdivmod__", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Float.__rfloordiv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__round__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Float.__rtruediv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__sizeof__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Float.__trunc__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Float.as_integer_ratio", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Float.is_integer", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Float.__pow__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__rpow__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__iadd__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__isub__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__imul__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__imod__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__ipow__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__pos__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.conjugate", "syft.lib.python.Float", Float),
        # String Methods
        ("syft.lib.python.String.__add__", "syft.lib.python.String", String),
        ("syft.lib.python.String.__contains__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.__eq__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.__float__", "syft.lib.python.Float", Float),
        ("syft.lib.python.String.__ge__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.__getitem__", "syft.lib.python.String", String),
        ("syft.lib.python.String.__gt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.__int__", "syft.lib.python.Int", Int),
        ("syft.lib.python.String.__iter__", "syft.lib.python.Any", Any),
        ("syft.lib.python.String.__le__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.__len__", "syft.lib.python.Int", Int),
        ("syft.lib.python.String.__lt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.__mod__", "syft.lib.python.String", String),
        ("syft.lib.python.String.__mul__", "syft.lib.python.String", String),
        ("syft.lib.python.String.__ne__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.__reversed__", "syft.lib.python.String", String),
        ("syft.lib.python.String.__sizeof__", "syft.lib.python.Int", Int),
        ("syft.lib.python.String.__str__", "syft.lib.python.String", String),
        ("syft.lib.python.String.capitalize", "syft.lib.python.String", String),
        ("syft.lib.python.String.casefold", "syft.lib.python.String", String),
        ("syft.lib.python.String.center", "syft.lib.python.String", String),
        ("syft.lib.python.String.count", "syft.lib.python.Int", Int),
        ("syft.lib.python.String.encode", "syft.lib.python.String", String),
        ("syft.lib.python.String.expandtabs", "syft.lib.python.String", String),
        ("syft.lib.python.String.find", "syft.lib.python.Int", Int),
        ("syft.lib.python.String.format", "syft.lib.python.String", String),
        ("syft.lib.python.String.format_map", "syft.lib.python.String", String),
        ("syft.lib.python.String.index", "syft.lib.python.Int", Int),
        ("syft.lib.python.String.isalnum", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isalpha", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isdecimal", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isdigit", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isidentifier", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.islower", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isnumeric", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isprintable", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isspace", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isupper", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.join", "syft.lib.python.String", String),
        ("syft.lib.python.String.ljust", "syft.lib.python.String", String),
        ("syft.lib.python.String.lower", "syft.lib.python.String", String),
        ("syft.lib.python.String.lstrip", "syft.lib.python.String", String),
        ("syft.lib.python.String.partition", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.String.replace", "syft.lib.python.String", String),
        ("syft.lib.python.String.rfind", "syft.lib.python.Int", Int),
        ("syft.lib.python.String.rindex", "syft.lib.python.Int", Int),
        ("syft.lib.python.String.rjust", "syft.lib.python.String", String),
        ("syft.lib.python.String.rpartition", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.String.rsplit", "syft.lib.python.List", List),
        ("syft.lib.python.String.rstrip", "syft.lib.python.String", String),
        ("syft.lib.python.String.split", "syft.lib.python.List", List),
        ("syft.lib.python.String.splitlines", "syft.lib.python.List", List),
        ("syft.lib.python.String.startswith", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.strip", "syft.lib.python.String", String),
        ("syft.lib.python.String.swapcase", "syft.lib.python.String", String),
        ("syft.lib.python.String.title", "syft.lib.python.String", String),
        ("syft.lib.python.String.translate", "syft.lib.python.String", String),
        ("syft.lib.python.String.upper", "syft.lib.python.String", String),
        ("syft.lib.python.String.zfill", "syft.lib.python.String", String),
        ("syft.lib.python.String.__contains__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.__rmul__", "syft.lib.python.String", String),
        ("syft.lib.python.String.endswith", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.isascii", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.String.istitle", "syft.lib.python.Bool", Bool),
        # Dict methods
        ("syft.lib.python.Dict.__contains__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Dict.__eq__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Dict.__format__", "syft.lib.python.String", String),
        ("syft.lib.python.Dict.__ge__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Dict.__getitem__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Dict.__gt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Dict.__iter__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Dict.__le__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Dict.__len__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Dict.__lt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Dict.__ne__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Dict.__sizeof__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Dict.__str__", "syft.lib.python.String", String),
        ("syft.lib.python.Dict.copy", "syft.lib.python.Dict", Dict),
        ("syft.lib.python.Dict.fromkeys", "syft.lib.python.Dict", Dict),
        # TODO: name conflict with syft.get()
        # ("syft.lib.python.Dict.get", "syft.lib.python.Any", Any),
        ("syft.lib.python.Dict.items", "syft.lib.python.List", List),
        ("syft.lib.python.Dict.keys", "syft.lib.python.List", List),
        ("syft.lib.python.Dict.pop", "syft.lib.python.Any", Any),
        ("syft.lib.python.Dict.popitem", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Dict.setdefault", "syft.lib.python.Any", Any),
        ("syft.lib.python.Dict.values", "syft.lib.python.List", List),
        # Int methods - subject to further change
        ("syft.lib.python.Int.__add__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__truediv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int.__divmod__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int.__floordiv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int.__invert__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__abs__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__bool__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Int.__divmod__", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Int.__rdivmod__", "syft.lib.python.Int", Tuple),
        ("syft.lib.python.Int.__radd__", "syft.lib.python.Int", Any),
        ("syft.lib.python.Int.__sub__", "syft.lib.python.Int", Any),
        ("syft.lib.python.Int.__rsub__", "syft.lib.python.Int", Any),
        ("syft.lib.python.Int.__rtruediv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__mul__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rmul__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__ceil__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__eq__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Int.__float__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int.__floor__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__floordiv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rfloordiv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__truediv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int.__mod__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rmod__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__pow__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rpow__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__lshift__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rlshift__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__round__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rshift__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rrshift__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__and__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rand__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__xor__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__xor__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__rxor__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__or__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__ror__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__ge__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Int.__lt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Int.__le__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Int.__gt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Int.__iadd__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__isub__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__imul__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__ifloordiv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__itruediv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__imod__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__ipow__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__ne__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Int.__neg__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__pos__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.as_integer_ratio", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Int.bit_length", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.denominator", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.from_bytes", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.real", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.imag", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.numerator", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.conjugate", "syft.lib.python.Int", Int),
        ("syft.lib.python.Int.__trunc__", "syft.lib.python.Int", Int),
        # Tuple
        ("syft.lib.python.Tuple.__add__", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Tuple.__contains__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Tuple.__ne__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Tuple.__ge__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Tuple.__gt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Tuple.__le__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Tuple.__lt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.Tuple.__mul__", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Tuple.__rmul__", "syft.lib.python.Tuple", Tuple),
        ("syft.lib.python.Tuple.__len__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Tuple.__getitem__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Tuple.count", "syft.lib.python.Int", Int),
        ("syft.lib.python.Tuple.index", "syft.lib.python.Int", Int),
        ("syft.lib.python.Tuple.__iter__", "syft.lib.python.Any", Any),
        # PyContainer - quite there
        ("syft.lib.python.Any.__add__", "syft.lib.python.PyContainer", Any),
        ("syft.lib.python.Any.__iter__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__next__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__radd__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__truediv__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__rtruediv__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__floordiv__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__rfloordiv__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__mul__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__rmul__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__sub__", "syft.lib.python.Any", Any),
        ("syft.lib.python.Any.__rsub__", "syft.lib.python.Any", Any),
        # ValueIndicies
        (
            "syft.lib.python.ValuesIndices.values",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.indices",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.eigenvalues",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.eigenvectors",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.solution",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.QR",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.sign",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.logabsdet",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.Q",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.R",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.LU",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.cloned_coefficient",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.U",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.S",
            "torch.Tensor",
            torch.Tensor,
        ),
        (
            "syft.lib.python.ValuesIndices.V",
            "torch.Tensor",
            torch.Tensor,
        ),
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
