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
from .float import Float
from .int import Int
from .list import List
from .none import SyNone  # noqa: F401
from .string import String


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


def add_primitives(ast: Globals, primitives: TypeList[Tuple[str, Any]]) -> None:
    for primitive, ref in primitives:
        parent = get_parent(primitive, ast)
        attr_name = primitive.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Class(
                attr_name,
                primitive,
                ref,
                return_type_name=primitive,
            ),
        )


def create_python_ast() -> Globals:
    ast = Globals()

    modules = [
        "syft",
        "syft.lib",
        "syft.lib.python",
    ]
    primitives = [
        ("syft.lib.python.Bool", Bool),
        ("syft.lib.python.Complex", Complex),
        ("syft.lib.python.Int", Int),
        ("syft.lib.python.Float", Float),
        ("syft.lib.python.List", List),
        ("syft.lib.python.String", String),
    ]

    add_modules(ast, modules)
    add_primitives(ast, primitives)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
