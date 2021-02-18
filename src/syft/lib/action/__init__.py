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
from syft.core.node.common.plan.plan import Plan

# syft relative
from . import action_wrapper  # noqa: 401


def create_plan_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client)

    modules = [
        "syft",
        "syft.core",
        "syft.core.node",
        "syft.core.node.common",
        "syft.core.node.common.action",
        "syft.core.node.common.action.get_object_action",
        "syft.core.node.common.plan",
        "syft.core.node.common.plan.plan",
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        (
            "syft.core.node.common.action.get_object_action.GetObjectAction",
            "syft.core.node.common.action.get_object_action.GetObjectAction",
            GetObjectAction,
        ),
        (
            "syft.core.node.common.plan.plan.Plan",
            "syft.core.node.common.plan.plan.Plan",
            Plan,
        ),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        ("syft.core.node.common.plan.plan.Plan.execute", "syft.lib.python._SyNone"),
        # ("syft.core.node.common.action.get_object_action.GetObjectAction.execute_action", "syft.lib.python._SyNone"),
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
