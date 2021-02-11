# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# syft absolute
from syft.ast import add_classes
from syft.ast import add_methods
from syft.ast import add_modules
from syft.ast.globals import Globals
from syft.core.node.common.action.get_object_action import GetObjectAction


def create_action_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client)

    modules = [
        "syft",
        "syft.core",
        "syft.core.node",
        "syft.core.node.common",
        "syft.core.node.common.action",
        "syft.core.node.common.action.get_object_action",
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        (
            "syft.core.node.common.action.get_object_action.GetObjectAction",
            "syft.core.node.common.action.get_object_action.GetObjectAction",
            GetObjectAction,
        ),
    ]

    methods: TypeList[TypeTuple[str, str]] = []

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
