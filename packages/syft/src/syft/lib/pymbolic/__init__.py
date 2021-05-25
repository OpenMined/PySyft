# stdlib
import functools
from typing import Any as TypeAny

# third party
import pymbolic as pmbl

# syft relative
from . import primitives  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "pymbolic"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "python": {"max_version": (3, 9, 99)},
}


def create_ast(client: TypeAny) -> Globals:
    ast = Globals(client=client)

    modules = ["pymbolic", "pymbolic.primitives"]
    classes = [
        (
            "pymbolic.primitives.Variable",
            "pymbolic.primitives.Variable",
            pmbl.primitives.Variable,
        ),
        (
            "pymbolic.primitives.Product",
            "pymbolic.primitives.Product",
            pmbl.primitives.Product,
        ),
    ]

    methods = [
        # Variable
        # ("pymbolic.primitives.Variable.name", "syft.lib.python.String"), # object attr
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
