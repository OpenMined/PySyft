"""Client AST creation.
"""
# stdlib
from typing import Any
from typing import Optional

# relative
from ...common.uid import UID
from ...node.abstract.node import AbstractNodeClient


def create_client_ast(client: Optional[AbstractNodeClient] = None) -> Any:
    # relative
    from ....ast import add_classes
    from ....ast import add_modules
    from ....ast.globals import Globals
    from .client import Client

    ast = Globals(client)

    modules = [
        "syft",
        "syft.core",
        "syft.core.common",
        "syft.core.common.uid",
        "syft.core.node",
        "syft.core.node.common",
        "syft.core.node.common.client",
    ]

    classes = [
        (
            "syft.core.node.common.client.Client",
            "syft.core.node.common.client.Client",
            Client,
        ),
        (
            "syft.core.common.uid.UID",
            "syft.core.common.uid.UID",
            UID,
        ),
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast
