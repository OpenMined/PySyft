# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import opacus

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "opacus"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "torch": {"min_version": "1.6.0"},
    "python": {"min_version": (3, 6, 9), "max_version": (3, 9, 99)},
}


def create_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("opacus", opacus),
        ("opacus.privacy_engine", opacus.privacy_engine),
    ]
    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        (
            "opacus.privacy_engine.PrivacyEngine",
            "opacus.privacy_engine.PrivacyEngine",
            opacus.privacy_engine.PrivacyEngine,
        ),
    ]

    methods = [
        (
            "opacus.privacy_engine.PrivacyEngine.to",
            "opacus.privacy_engine.PrivacyEngine",
        ),
        ("opacus.privacy_engine.PrivacyEngine.step", "syft.lib.python._SyNone"),
        ("opacus.privacy_engine.PrivacyEngine.zero_grad", "syft.lib.python._SyNone"),
        ("opacus.privacy_engine.PrivacyEngine.zero_grad", "syft.lib.python._SyNone"),
        (
            "opacus.privacy_engine.PrivacyEngine.get_privacy_spent",
            "syft.lib.python.Tuple",
        ),
        ("opacus.privacy_engine.PrivacyEngine.attach", "syft.lib.python._SyNone"),
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
