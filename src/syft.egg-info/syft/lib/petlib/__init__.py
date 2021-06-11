# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import petlib as pl
import zksk as zk  # noqa: 401 # required for pl.ec and pl.bn

# syft relative
from . import bn  # noqa: 401
from . import ecpt  # noqa: 401
from . import ecpt_group  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "petlib"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "python": {"max_version": (3, 9, 99)},
}


def create_ast(client: TypeAny) -> Globals:
    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("petlib", pl),
        ("petlib.ec", pl.ec),
        ("petlib.bn", pl.bn),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("petlib.ec.EcPt", "petlib.ec.EcPt", pl.ec.EcPt),
        ("petlib.ec.EcGroup", "petlib.ec.EcGroup", pl.ec.EcGroup),
        ("petlib.bn.Bn", "petlib.bn.Bn", pl.bn.Bn),
    ]

    methods = [
        ("petlib.ec.EcPt.group", "petlib.ec.EcGroup"),
        ("petlib.ec.EcPt.__copy__", "petlib.ec.EcPt"),
        ("petlib.ec.EcPt.__add__", "petlib.ec.EcPt"),
        ("petlib.ec.EcPt.pt_add", "petlib.ec.EcPt"),
        ("petlib.ec.EcPt.pt_add_inplace", "petlib.ec.EcPt"),
        ("petlib.ec.EcPt.pt_mul", "petlib.ec.EcPt"),
        ("petlib.ec.EcPt.pt_mul_inplace", "petlib.ec.EcPt"),
        ("petlib.ec.EcPt.__rmul__", "petlib.ec.EcPt"),
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
