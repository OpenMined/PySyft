# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
# import tenseal

# syft relative
from ...ast.globals import Globals
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules


def create_tenseal_ast() -> Globals:
    ast = Globals()

    modules = [
        "tenseal",
        "tenseal.version",
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = []

    methods = [
        ("tenseal.version.__version__", "syft.lib.python.String"),
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
