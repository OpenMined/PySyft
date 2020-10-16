# stdlib
from typing import Any
from typing import List as TypeList
from typing import Tuple

# syft relative
from ...ast.globals import Globals
from ...ast.klass import Class
from ...ast.module import Module
from .bool import Bool
from .complex import Complex
from .dict import Dict
from .float import Float
from .int import Int
from .list import List
from .none import SyNone
from .string import String
from .primitive_interface import PyPrimitive
from .primitive_container import PyContainer
from .tuple import Tuple as SyTuple

def get_parent(path: str, root: Any) -> Module:
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


def add_classes(ast: Globals, paths: TypeList[Tuple[str, str, Any]]) -> None:
    for path, return_type, ref in paths:
        parent = get_parent(path, ast)
        attr_name = path.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Class(
                attr_name,
                path,
                ref,
                return_type_name=return_type,
            ),
        )


def add_methods(ast: Globals, paths: TypeList[Tuple[str, str, Any]]) -> None:
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
        ("syft.lib.python.SyNone", "syft.lib.python.SyNone", SyNone),
        ("syft.lib.python.PyPrimitive", "syft.lib.python.PyPrimitive", PyPrimitive),
        ("syft.lib.python.PyContainer", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.Tuple", "syft.lib.python.Tuple", SyTuple)
    ]

    methods = [
        #List methods - quite there
        ("syft.lib.python.List.__len__", "syft.lib.python.Int", List),
        ("syft.lib.python.List.__getitem__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.List.__iter__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.List.__next__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.List.__add__", "syft.lib.python.List", List),
        ("syft.lib.python.List.append", "syft.lib.python.SyNone", SyNone),
        ("syft.lib.python.List.__gt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__lt__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__le__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__ge__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__iadd__", "syft.lib.python.List", List),
        ("syft.lib.python.List.__imul__", "syft.lib.python.List", List),
        ("syft.lib.python.List.__iadd__", "syft.lib.python.List", List),
        ("syft.lib.python.List.__contains__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__delattr__", "syft.lib.python.None", SyNone),
        ("syft.lib.python.List.__delitem__", "syft.lib.python.None", SyNone),
        ("syft.lib.python.List.__eq__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__mul__", "syft.lib.python.List", List),
        ("syft.lib.python.List.__ne__", "syft.lib.python.Bool", Bool),
        ("syft.lib.python.List.__sizeof__", "syft.lib.python.Int", Int),
        ("syft.lib.python.List.__len__", "syft.lib.python.Int", Int),
        ("syft.lib.python.List.__getitem__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.List.copy", "syft.lib.python.List", List),
        ("syft.lib.python.List.count", "syft.lib.python.Int", Int),


        #Bool methods - quite there
        ("syft.lib.python.Bool.__abs__", "syft.lib.python.Int", Int),
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
        #not sure about this
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
        #not sure about this
        ("syft.lib.python.Bool.__rsub__", "syft.lib.python.Int", Int),
        #not sure about this
        ("syft.lib.python.Bool.__rtruediv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__rxor__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__sub__", "syft.lib.python.Int", Int),
        #not sure about this
        ("syft.lib.python.Bool.__truediv__", "syft.lib.python.Int", Int),
        ("syft.lib.python.Bool.__xor__", "syft.lib.python.Int", Int),

        #Float methods - subject to further change due
        ("syft.lib.python.Float.__add__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__truediv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__divmod__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Float.__floordiv__", "syft.lib.python.Float", Float),

        #Int methods - subject to further change
        ("syft.lib.python.Int.__add__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int.__truediv__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int.__divmod__", "syft.lib.python.Float", Float),
        ("syft.lib.python.Int.__floordiv__", "syft.lib.python.Float", Float),

        #PyContainer - quite there
        ("syft.lib.python.PyContainer.__add__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__iter__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__next__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__radd__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__truediv__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__rtruediv__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__floordiv__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__rfloordiv__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__mul__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__rmul__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__sub__", "syft.lib.python.PyContainer", PyContainer),
        ("syft.lib.python.PyContainer.__rsub__", "syft.lib.python.PyContainer", PyContainer)
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
