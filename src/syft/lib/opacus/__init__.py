# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import opacus

# syft relative
from ...ast.globals import Globals
from ...ast.klass import Class
from ...ast.module import Module


def get_parent(path: str, root: TypeAny) -> Module:
    parent = root
    for step in path.split(".")[:-1]:
        parent = parent.attrs[step]
    return parent


def add_modules(ast: Globals, modules: TypeList[str]) -> None:
    for module in modules:
        parent = get_parent(module, ast)
        attr_name = module.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Module(
                attr_name,
                module,
                None,
                return_type_name="",
            ),
        )


def add_classes(ast: Globals, paths: TypeList[TypeTuple[str, str, TypeAny]]) -> None:
    for path, return_type, ref in paths:
        parent = get_parent(path, ast)
        attr_name = path.rsplit(".", 1)[-1]

        parent.add_attr(
            attr_name=attr_name,
            attr=Class(
                attr_name,
                path,
                ref,
                return_type_name=return_type,
            ),
        )


def add_methods(ast: Globals, paths: TypeList[TypeTuple[str, str]]) -> None:
    for path, return_type in paths:
        parent = get_parent(path, ast)
        path_list = path.split(".")
        parent.add_path(
            path=path_list, index=len(path_list) - 1, return_type_name=return_type
        )


def create_ast() -> Globals:
    ast = Globals()

    modules = [
        "opacus",
        "opacus.privacy_engine",
    ]
    classes = [
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
        ),  # TODO: This is a Tuple
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
