"""add the sympc library into syft."""

# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# syft relative
from . import rst_share  # noqa: 401
from . import session  # noqa: 401
from . import share  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "sympc"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "torch": {"min_version": "1.6.0", "max_version": "1.8.1"},
    "python": {"min_version": (3, 7), "max_version": (3, 9, 99)},
}


def create_ast(client: TypeAny = None) -> Globals:
    """Add the modules, classes and attributes from sympc to syft.

    Args:
        client: Client

    Returns:
        Globals

    """
    # third party
    import sympc

    # syft relative
    from . import rst_share  # noqa: 401
    from . import session  # noqa: 401
    from . import share  # noqa: 401

    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = sympc.api.allowed_external_modules
    add_modules(ast, modules)

    classes: TypeList[TypeTuple[str, str, TypeAny]] = sympc.api.allowed_external_classes
    add_classes(ast, classes)

    attrs: TypeList[TypeTuple[str, str]] = sympc.api.allowed_external_attrs
    add_methods(ast, attrs)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
