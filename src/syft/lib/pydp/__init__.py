# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
from pydp.algorithms.laplacian import BoundedMean
from pydp.algorithms.laplacian import BoundedStandardDeviation
from pydp.algorithms.laplacian import BoundedSum
from pydp.algorithms.laplacian import BoundedVariance
from pydp.algorithms.laplacian import Count
from pydp.algorithms.laplacian import Max
from pydp.algorithms.laplacian import Median
from pydp.algorithms.laplacian import Min
from pydp.algorithms.laplacian import Percentile

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
        (
            "pydp.algorithms.laplacian.BoundedSum",
            "pydp.algorithms.laplacian.BoundedSum",
            BoundedSum,
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation",
            "pydp.algorithms.laplacian.BoundedStandardDeviation",
            BoundedStandardDeviation,
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance",
            "pydp.algorithms.laplacian.BoundedVariance",
            BoundedVariance,
        ),
        (
            "pydp.algorithms.laplacian.Min",
            "pydp.algorithms.laplacian.Min",
            Min,
        ),
        (
            "pydp.algorithms.laplacian.Max",
            "pydp.algorithms.laplacian.Max",
            Max,
        ),
        (
            "pydp.algorithms.laplacian.Median",
            "pydp.algorithms.laplacian.Median",
            Median,
        ),
        (
            "pydp.algorithms.laplacian.Percentile",
            "pydp.algorithms.laplacian.Percentile",
            Percentile,
        ),
        (
            "pydp.algorithms.laplacian.Count",
            "pydp.algorithms.laplacian.Count",
            Count,
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
        (
            "pydp.algorithms.laplacian.BoundedSum.quick_result",
            "syft.lib.python.Float",
            BoundedSum.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.add_entries",
            "syft.lib.python._SyNone",
            BoundedSum.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.add_entry",
            "syft.lib.python._SyNone",
            BoundedSum.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.privacy_budget_left",
            "syft.lib.python.Float",
            BoundedSum.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.reset",
            "syft.lib.python._SyNone",
            BoundedSum.reset,
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            BoundedSum.result,
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.noise_confidence_interval",
            "syft.lib.python.Float",
            BoundedSum.noise_confidence_interval,
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.quick_result",
            "syft.lib.python.Float",
            BoundedStandardDeviation.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.add_entries",
            "syft.lib.python._SyNone",
            BoundedStandardDeviation.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.add_entry",
            "syft.lib.python._SyNone",
            BoundedStandardDeviation.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.privacy_budget_left",
            "syft.lib.python.Float",
            BoundedStandardDeviation.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.reset",
            "syft.lib.python._SyNone",
            BoundedStandardDeviation.reset,
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            BoundedStandardDeviation.result,
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.noise_confidence_interval",
            "syft.lib.python.Float",
            BoundedStandardDeviation.noise_confidence_interval,
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.quick_result",
            "syft.lib.python.Float",
            BoundedVariance.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.add_entries",
            "syft.lib.python._SyNone",
            BoundedVariance.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.add_entry",
            "syft.lib.python._SyNone",
            BoundedVariance.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.privacy_budget_left",
            "syft.lib.python.Float",
            BoundedVariance.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.reset",
            "syft.lib.python._SyNone",
            BoundedVariance.reset,
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            BoundedVariance.result,
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.noise_confidence_interval",
            "syft.lib.python.Float",
            BoundedVariance.noise_confidence_interval,
        ),
        (
            "pydp.algorithms.laplacian.Min.quick_result",
            "syft.lib.python.Float",
            Min.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.Min.add_entries",
            "syft.lib.python._SyNone",
            Min.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.Min.add_entry",
            "syft.lib.python._SyNone",
            Min.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.Min.privacy_budget_left",
            "syft.lib.python.Float",
            Min.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.Min.reset",
            "syft.lib.python._SyNone",
            Min.reset,
        ),
        (
            "pydp.algorithms.laplacian.Min.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            Min.result,
        ),
        (
            "pydp.algorithms.laplacian.Min.noise_confidence_interval",
            "syft.lib.python.Float",
            Min.noise_confidence_interval,
        ),
        (
            "pydp.algorithms.laplacian.Max.quick_result",
            "syft.lib.python.Float",
            Max.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.Max.add_entries",
            "syft.lib.python._SyNone",
            Max.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.Max.add_entry",
            "syft.lib.python._SyNone",
            Max.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.Max.privacy_budget_left",
            "syft.lib.python.Float",
            Max.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.Max.reset",
            "syft.lib.python._SyNone",
            Max.reset,
        ),
        (
            "pydp.algorithms.laplacian.Max.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            Max.result,
        ),
        (
            "pydp.algorithms.laplacian.Max.noise_confidence_interval",
            "syft.lib.python.Float",
            Max.noise_confidence_interval,
        ),
        (
            "pydp.algorithms.laplacian.Median.quick_result",
            "syft.lib.python.Float",
            Median.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.Median.add_entries",
            "syft.lib.python._SyNone",
            Median.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.Median.add_entry",
            "syft.lib.python._SyNone",
            Median.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.Median.privacy_budget_left",
            "syft.lib.python.Float",
            Median.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.Median.reset",
            "syft.lib.python._SyNone",
            Median.reset,
        ),
        (
            "pydp.algorithms.laplacian.Median.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            Median.result,
        ),
        (
            "pydp.algorithms.laplacian.Median.noise_confidence_interval",
            "syft.lib.python.Float",
            Median.noise_confidence_interval,
        ),
        (
            "pydp.algorithms.laplacian.Percentile.quick_result",
            "syft.lib.python.Float",
            Percentile.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.Percentile.add_entries",
            "syft.lib.python._SyNone",
            Percentile.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.Percentile.add_entry",
            "syft.lib.python._SyNone",
            Percentile.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.Percentile.privacy_budget_left",
            "syft.lib.python.Float",
            Percentile.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.Percentile.reset",
            "syft.lib.python._SyNone",
            Percentile.reset,
        ),
        (
            "pydp.algorithms.laplacian.Percentile.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            Percentile.result,
        ),
        (
            "pydp.algorithms.laplacian.Percentile.noise_confidence_interval",
            "syft.lib.python.Float",
            Percentile.noise_confidence_interval,
        ),
        (
            "pydp.algorithms.laplacian.Count.quick_result",
            "syft.lib.python.Float",
            Count.quick_result,
        ),
        (
            "pydp.algorithms.laplacian.Count.add_entries",
            "syft.lib.python._SyNone",
            Count.add_entries,
        ),
        (
            "pydp.algorithms.laplacian.Count.add_entry",
            "syft.lib.python._SyNone",
            Count.add_entry,
        ),
        (
            "pydp.algorithms.laplacian.Count.privacy_budget_left",
            "syft.lib.python.Float",
            Count.privacy_budget_left,
        ),
        (
            "pydp.algorithms.laplacian.Count.reset",
            "syft.lib.python._SyNone",
            Count.reset,
        ),
        (
            "pydp.algorithms.laplacian.Count.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
            Count.result,
        ),
        (
            "pydp.algorithms.laplacian.Count.noise_confidence_interval",
            "syft.lib.python.Float",
            Count.noise_confidence_interval,
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
