# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
from pydp.algorithms.laplacian import BoundedMean

# syft relative
from ...ast.globals import Globals
from ...ast.klass import Class
from ...ast.module import Module
from ..misc.union import UnionGenerator

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


def add_methods(ast: Globals, paths: TypeList[TypeTuple[str, str, TypeAny]]) -> None:
    for path, return_type, _ in paths:
        parent = get_parent(path, ast)
        path_list = path.split(".")
        parent.add_path(
            path=path_list, index=len(path_list) - 1, return_type_name=return_type
        )


def create_pydp_ast() -> Globals:
    ast = Globals()

    modules = [
        "pydp",
        "pydp.algorithms",
        "pydp.algorithms.laplacian",
    ]
    classes = [
        (
            "pydp.algorithms.laplacian.BoundedMean",
            "pydp.algorithms.laplacian.BoundedMean",
            BoundedMean,
        ),
    ]

    methods = [
        (
            "pydp.algorithms.laplacian.BoundedMean.quick_result",
            "syft.lib.python.Float",
            BoundedMean.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.add_entries",
            "syft.lib.python._SyNone",
            BoundedMean.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.add_entry",
            "syft.lib.python._SyNone",
            BoundedMean.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.epsilon",
            "syft.lib.python.Float",
            BoundedMean.epsilon,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.l0_sensitivity",
            "syft.lib.python.Int",
            BoundedMean.l0_sensitivity,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.linf_sensitivity",
            "syft.lib.python.Int",
            BoundedMean.linf_sensitivity,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.privacy_budget_left",
            "syft.lib.python.Float",
            BoundedMean.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.reset",
            "syft.lib.python._SyNone",
            BoundedMean.reset,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            BoundedMean.result,
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.noise_confidence_interval",
            "syft.lib.python.Float",
            BoundedMean.noise_confidence_interval,
        ),
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
