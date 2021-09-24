"""SMPC AST Creation.

__init__ file for SMPC. This defines various modules, classes and methods which we currently support.
We create an AST for all these modules, classes and methods so that they can be called remotely.
"""
# stdlib
from typing import Optional

# relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ...core.node.abstract.node import AbstractNodeClient
from .store import CryptoStore

LIB_NAME = "SMPC"
PACKAGE_SUPPORT = {"lib": LIB_NAME}


def create_smpc_ast(client: Optional[AbstractNodeClient] = None) -> Globals:
    ast = Globals(client)

    modules = ["syft", "syft.lib", "syft.lib.SMPC", "syft.lib.SMPC.store"]

    classes = [
        (
            "syft.lib.SMPC.store.CryptoStore",
            "syft.lib.SMPC.store.CryptoStore",
            CryptoStore,
        ),
    ]

    methods = [
        (
            "syft.lib.SMPC.store.CryptoStore.get_primitives_from_store",
            "syft.lib.python.List",
        ),
        ("syft.lib.SMPC.store.CryptoStore.store", "syft.lib.python.Dict"),
        (
            "syft.lib.SMPC.store.CryptoStore.populate_store",
            "syft.lib.python._SyNone",
        ),
        (
            "syft.lib.SMPC.store.CryptoStore.print",
            "syft.lib.python._SyNone",
        ),
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast
