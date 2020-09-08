# stdlib
from typing import Any
from typing import List
from typing import Tuple

# syft relative
from ...ast.globals import Globals
from ...ast.module import Module
from ...ast.klass import Class
from .bool import Bool
from .int import Int


def get_parent(path: str, root: Module) -> Module:
    parent = root
    for step in path.split(".")[:-1]:
        parent = parent.attrs[step]
    return parent


def add_modules(ast: Globals, modules: List[str]) -> None:
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


def add_primitives(ast: Globals, primitives: List[Tuple[str, Any]]):
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
        ("syft.lib.python.Int", Int),
        ("syft.lib.python.Bool", Bool),
    ]

    add_modules(ast, modules)
    add_primitives(ast, primitives)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
