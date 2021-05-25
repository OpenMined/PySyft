# stdlib
from typing import Optional

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ...core.node.abstract.node import AbstractNodeClient
from .scalar import GammaScalar  # 401
from .scalar import IntermediatePhiScalar  # 401
from .scalar import PhiScalar  # 401
from .scalar import Scalar  # 401


def create_adp_ast(client: Optional[AbstractNodeClient] = None) -> Globals:
    ast = Globals(client)

    modules = ["syft", "syft.lib", "syft.lib.adp"]
    classes = [
        ("syft.lib.adp.GammaScalar", "syft.lib.adp.Scalar", GammaScalar),
        ("syft.lib.adp.Scalar", "syft.lib.adp.Scalar", Scalar),
        ("syft.lib.adp.PhiScalar", "syft.lib.adp.PhiScalar", PhiScalar),
        (
            "syft.lib.adp.IntermediatePhiScalar",
            "syft.lib.adp.IntermediatePhiScalar",
            IntermediatePhiScalar,
        ),
    ]

    methods = [
        # Scalar
        ("syft.lib.adp.Scalar.max_val", "syft.lib.python.Float"),
        ("syft.lib.adp.Scalar.min_val", "syft.lib.python.Float"),
        ("syft.lib.adp.Scalar.value", "syft.lib.python.Float"),
        ("syft.lib.adp.Scalar.publish", "syft.lib.python.Float"),
        # PhiScalar
        (
            "syft.lib.adp.PhiScalar.__add__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.PhiScalar.__mul__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.PhiScalar.__radd__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.PhiScalar.__rmul__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.PhiScalar.__rsub__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.PhiScalar.__sub__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        ("syft.lib.adp.PhiScalar.gamma", "syft.lib.adp.GammaScalar"),
        (
            "syft.lib.adp.PhiScalar.input_entities",
            "syft.lib.python.List",
        ),  # TypeList[Entity]
        (
            "syft.lib.adp.PhiScalar.input_polys",
            "syft.lib.python.Set",
        ),  # TypeSet[BasicSymbol]
        (
            "syft.lib.adp.PhiScalar.input_scalars",
            "syft.lib.python.List",
        ),  # TypeList[Union[PhiScalar, GammaScalar]]
        ("syft.lib.adp.PhiScalar.max_lipschitz", "syft.lib.python.Float"),
        ("syft.lib.adp.PhiScalar.max_lipschitz_wrt_entity", "syft.lib.python.Float"),
        ("syft.lib.adp.PhiScalar.max_val", "syft.lib.python.Float"),
        ("syft.lib.adp.PhiScalar.min_val", "syft.lib.python.Float"),
        ("syft.lib.adp.PhiScalar.value", "syft.lib.python.Float"),
        ("syft.lib.adp.PhiScalar.publish", "syft.lib.python.Float"),
        # IntermediatePhiScalar
        ("syft.lib.adp.IntermediatePhiScalar.gamma", "syft.lib.adp.GammaScalar"),
        (
            "syft.lib.adp.IntermediatePhiScalar.__add__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.IntermediatePhiScalar.__mul__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.IntermediatePhiScalar.__radd__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.IntermediatePhiScalar.__rmul__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.IntermediatePhiScalar.__rsub__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.lib.adp.IntermediatePhiScalar.__sub__",
            "syft.lib.adp.IntermediatePhiScalar",
        ),  # Union?
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast
