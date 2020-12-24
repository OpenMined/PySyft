# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple
from typing import Union as TypeUnion

# third party
import opacus

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals

LIB_NAME = "opacus"
PACKAGE_SUPPORT = {"lib": LIB_NAME}


def update_ast(ast: TypeUnion[Globals, TypeAny]) -> None:
    opacus_ast = create_ast()
    ast.add_attr(attr_name=LIB_NAME, attr=opacus_ast.attrs[LIB_NAME])


def create_ast() -> Globals:
    ast = Globals()

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
        (
            "opacus.privacy_engine.PrivacyEngine.get_privacy_spent",
            "syft.lib.python.List",
        ),
        ("opacus.privacy_engine.PrivacyEngine.attach", "syft.lib.python._SyNone"),
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
