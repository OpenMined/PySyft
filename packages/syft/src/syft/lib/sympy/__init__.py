# stdlib
import functools
from typing import Any as TypeAny

# third party
import sympy as sym

# syft relative
from . import core  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "sympy"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "python": {"max_version": (3, 9, 99)},
}


def create_ast(client: TypeAny) -> Globals:
    ast = Globals(client=client)

    modules = [
        "sympy",
        "sympy.core",
        "sympy.core.mul",
        "sympy.core.add",
        "sympy.core.symbol",
    ]
    classes = [
        (
            "sympy.core.symbol.Symbol",
            "sympy.core.symbol.Symbol",
            sym.core.symbol.Symbol,
        ),
        (
            "sympy.core.mul.Mul",
            "sympy.core.mul.Mul",
            sym.core.mul.Mul,
        ),
        (
            "sympy.core.mul.Add",
            "sympy.core.mul.Add",
            sym.core.mul.Mul,
        ),
    ]

    methods = []  # type: ignore

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
