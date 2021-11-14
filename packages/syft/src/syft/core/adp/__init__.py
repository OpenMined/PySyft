# stdlib
from typing import Optional

# relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..node.abstract.node import AbstractNodeClient
from .scalar.abstract.scalar import Scalar  # noqa: 401
from .scalar.gamma_scalar import GammaScalar  # noqa: 401
from .scalar.intermediate_phi_scalar import IntermediatePhiScalar  # noqa: 401
from .scalar.phi_scalar import PhiScalar  # noqa: 401


def create_adp_ast(client: Optional[AbstractNodeClient] = None) -> Globals:
    ast = Globals(client)

    modules = ["syft", "syft.core", "syft.core.adp"]
    classes = [
        ("syft.core.adp.GammaScalar", "syft.core.adp.Scalar", GammaScalar),
        ("syft.core.adp.Scalar", "syft.core.adp.Scalar", Scalar),
        ("syft.core.adp.PhiScalar", "syft.core.adp.PhiScalar", PhiScalar),
        (
            "syft.core.adp.IntermediatePhiScalar",
            "syft.core.adp.IntermediatePhiScalar",
            IntermediatePhiScalar,
        ),
    ]

    methods = [
        # Scalar
        ("syft.core.adp.Scalar.max_val", "syft.lib.python.Float"),
        ("syft.core.adp.Scalar.min_val", "syft.lib.python.Float"),
        ("syft.core.adp.Scalar.value", "syft.lib.python.Float"),
        ("syft.core.adp.Scalar.publish", "syft.lib.python.Float"),
        # PhiScalar
        (
            "syft.core.adp.PhiScalar.__add__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.PhiScalar.__mul__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.PhiScalar.__radd__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.PhiScalar.__rmul__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.PhiScalar.__rsub__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.PhiScalar.__sub__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        ("syft.core.adp.PhiScalar.gamma", "syft.core.adp.GammaScalar"),
        (
            "syft.core.adp.PhiScalar.input_entities",
            "syft.lib.python.List",
        ),  # TypeList[Entity]
        (
            "syft.core.adp.PhiScalar.input_polys",
            "syft.lib.python.Set",
        ),  # TypeSet[BasicSymbol]
        (
            "syft.core.adp.PhiScalar.input_scalars",
            "syft.lib.python.List",
        ),  # TypeList[Union[PhiScalar, GammaScalar]]
        ("syft.core.adp.PhiScalar.max_lipschitz", "syft.lib.python.Float"),
        ("syft.core.adp.PhiScalar.max_lipschitz_wrt_entity", "syft.lib.python.Float"),
        ("syft.core.adp.PhiScalar.max_val", "syft.lib.python.Float"),
        ("syft.core.adp.PhiScalar.min_val", "syft.lib.python.Float"),
        ("syft.core.adp.PhiScalar.value", "syft.lib.python.Float"),
        ("syft.core.adp.PhiScalar.publish", "syft.lib.python.Float"),
        # IntermediatePhiScalar
        ("syft.core.adp.IntermediatePhiScalar.gamma", "syft.core.adp.GammaScalar"),
        (
            "syft.core.adp.IntermediatePhiScalar.__add__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.IntermediatePhiScalar.__mul__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.IntermediatePhiScalar.__radd__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.IntermediatePhiScalar.__rmul__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.IntermediatePhiScalar.__rsub__",
            "syft.core.adp.IntermediatePhiScalar",
        ),  # Union?
        (
            "syft.core.adp.IntermediatePhiScalar.__sub__",
            "syft.core.adp.IntermediatePhiScalar",
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
