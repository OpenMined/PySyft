# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import zksk as zk  # noqa: 401 # required for pl.ec and pl.bn

# syft relative
from . import nizk  # noqa: 401
from . import secret  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "zksk"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "python": {"max_version": (3, 9, 99)},
}


def create_ast(client: TypeAny) -> Globals:
    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("zksk", zk),
        ("zksk.utils", zk.utils),
        ("zksk.primitives", zk.primitives),
        ("zksk.primitives.dlrep", zk.primitives.dlrep),
        ("zksk.expr", zk.expr),
        ("zksk.base", zk.base),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        (
            "zksk.primitives.dlrep.DLRep",
            "zksk.primitives.dlrep.DLRep",
            zk.primitives.dlrep.DLRep,
        ),
        ("zksk.expr.Secret", "zksk.expr.Secret", zk.expr.Secret),
        ("zksk.expr.Expression", "zksk.expr.Expression", zk.expr.Expression),
        ("zksk.base.NIZK", "zksk.base.NIZK", zk.base.NIZK),
    ]

    methods = [
        ("zksk.utils.make_generators", "syft.lib.python.List"),
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
