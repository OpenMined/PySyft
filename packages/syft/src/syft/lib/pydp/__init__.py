# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import pydp
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
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..misc.union import UnionGenerator
from ..util import generic_update_ast

LIB_NAME = "pydp"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "python": {"max_version": (3, 8, 99)},
}


def create_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("pydp", pydp),
        ("pydp.algorithms", pydp.algorithms),
        ("pydp.algorithms.laplacian", pydp.algorithms.laplacian),
    ]
    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
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
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.quick_result",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.BoundedSum.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.quick_result",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.BoundedStandardDeviation.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.quick_result",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.BoundedVariance.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Min.quick_result",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Min.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Min.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Min.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Min.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Min.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.Min.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Max.quick_result",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Max.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Max.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Max.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Max.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Max.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.Max.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Median.quick_result",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Median.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Median.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Median.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Median.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Median.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.Median.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Percentile.quick_result",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Percentile.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Percentile.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Percentile.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Percentile.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Percentile.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.Percentile.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Count.quick_result",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Count.add_entries",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Count.add_entry",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Count.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.Count.reset",
            "syft.lib.python._SyNone",
        ),
        (
            "pydp.algorithms.laplacian.Count.result",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        (
            "pydp.algorithms.laplacian.Count.noise_confidence_interval",
            "syft.lib.python.Float",
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


update_ast = functools.partial(generic_update_ast, LIB_NAME, create_ast)
