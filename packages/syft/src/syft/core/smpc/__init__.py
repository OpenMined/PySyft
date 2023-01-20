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
from ..node.abstract.node import AbstractNodeClient
from .store import CryptoStore

from .protocol import beaver  # noqa: 401 isort: skip


def create_smpc_ast(client: Optional[AbstractNodeClient] = None) -> Globals:
    """Creates SMPC Abstract Syntax Tree (AST).
        Define a set of modules, classes and methods which are used for the creation of the SMPC AST.

    Args:
        client (Optional[AbstractNodeClient]):  Input client object for AST operations.

    Returns:
        The constructed SMPC AST.
    """
    ast = Globals(client)

    modules = [
        "syft",
        "syft.core",
        "syft.core.smpc",
        "syft.core.smpc.store",
        "syft.core.smpc.protocol",
        "syft.core.smpc.protocol.spdz",
        "syft.core.smpc.protocol.spdz.spdz",
    ]

    classes = [
        (
            "syft.core.smpc.store.CryptoStore",
            "syft.core.smpc.store.CryptoStore",
            CryptoStore,
        ),
    ]

    methods = [
        (
            "syft.core.smpc.store.CryptoStore.get_primitives_from_store",
            "syft.lib.python.List",
        ),
        ("syft.core.smpc.store.CryptoStore.store", "syft.lib.python.Dict"),
        (
            "syft.core.smpc.store.CryptoStore.populate_store",
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
