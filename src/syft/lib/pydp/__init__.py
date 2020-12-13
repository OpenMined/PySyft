# third party
import pydp

# syft relative
from ...ast.globals import Globals

from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules

from ..misc.union import UnionGenerator


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
            pydp.algorithms.laplacian.BoundedMean,
        ),
    ]

    methods = [
        (
            "pydp.algorithms.laplacian.BoundedMean.quick_result",
            UnionGenerator["syft.lib.python.Float", "syft.lib.python.Int"],
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.add_entries",
            "syft.lib.python._SyNone",
        ),
        ("pydp.algorithms.laplacian.BoundedMean.add_entry", "syft.lib.python._SyNone"),
        ("pydp.algorithms.laplacian.BoundedMean.memory_used", "syft.lib.python.Float"),
        (
            "pydp.algorithms.laplacian.BoundedMean.noise_confidence_interval",
            "syft.lib.python.Float",
        ),
        (
            "pydp.algorithms.laplacian.BoundedMean.privacy_budget_left",
            "syft.lib.python.Float",
        ),
        ("pydp.algorithms.laplacian.BoundedMean.reset", "syft.lib.python._SyNone"),
        (
            "pydp.algorithms.laplacian.BoundedMean.result",
            UnionGenerator["syft.lib.python.Float", "syft.lib.python.Int"],
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
