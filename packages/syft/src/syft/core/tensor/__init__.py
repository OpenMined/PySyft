# stdlib
from typing import Optional

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ...core.node.abstract.node import AbstractNodeClient
from .fixed_precision_tensor import FixedPrecisionTensor
from .smpc.share_tensor import ShareTensor
from .tensor import Tensor


def create_tensor_ast(client: Optional[AbstractNodeClient] = None) -> Globals:
    ast = Globals(client)

    modules = [
        "syft",
        "syft.core",
        "syft.core.tensor",
        "syft.core.tensor.fixed_precision_tensor",
        "syft.core.tensor.tensor",
        "syft.core.tensor.smpc",
        "syft.core.tensor.smpc.share_tensor",
    ]
    classes = [
        ("syft.core.tensor.tensor.Tensor", "syft.core.tensor.tensor.Tensor", Tensor),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
            ShareTensor,
        ),
        (
            "syft.core.tensor.fixed_precision_tensor.FixedPrecisionTensor",
            "syft.core.tensor.fixed_precision_tensor.FixedPrecisionTensor",
            FixedPrecisionTensor,
        ),
    ]

    methods = [
        # Tensor
        ("syft.core.tensor.tensor.Tensor.T", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__abs__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__add__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__divmod__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__eq__", "syft.core.tensor.tensor.Tensor"),
        (
            "syft.core.tensor.tensor.Tensor.__floordiv__",
            "syft.core.tensor.tensor.Tensor",
        ),
        ("syft.core.tensor.tensor.Tensor.__ge__", "syft.core.tensor.tensor.Tensor"),
        (
            "syft.core.tensor.tensor.Tensor.__getitem__",
            "syft.core.tensor.tensor.Tensor",
        ),
        ("syft.core.tensor.tensor.Tensor.__gt__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__index__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__invert__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__le__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__len__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__lshift__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__lt__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__matmul__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__mul__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__ne__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__neg__", "syft.core.tensor.tensor.Tensor"),
        (
            "syft.core.tensor.tensor.Tensor.__pos__",
            "syft.core.tensor.tensor.Tensor",
        ),  # useless?
        ("syft.core.tensor.tensor.Tensor.__pow__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__radd__", "syft.core.tensor.tensor.Tensor"),
        (
            "syft.core.tensor.tensor.Tensor.__rdivmod__",
            "syft.core.tensor.tensor.Tensor",
        ),
        (
            "syft.core.tensor.tensor.Tensor.__rfloordiv__",
            "syft.core.tensor.tensor.Tensor",
        ),
        (
            "syft.core.tensor.tensor.Tensor.__rlshift__",
            "syft.core.tensor.tensor.Tensor",
        ),
        (
            "syft.core.tensor.tensor.Tensor.__rmatmul__",
            "syft.core.tensor.tensor.Tensor",
        ),
        ("syft.core.tensor.tensor.Tensor.__rmul__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__rpow__", "syft.core.tensor.tensor.Tensor"),
        (
            "syft.core.tensor.tensor.Tensor.__rrshift__",
            "syft.core.tensor.tensor.Tensor",
        ),
        ("syft.core.tensor.tensor.Tensor.__rshift__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__rsub__", "syft.core.tensor.tensor.Tensor"),
        (
            "syft.core.tensor.tensor.Tensor.__rtruediv__",
            "syft.core.tensor.tensor.Tensor",
        ),
        ("syft.core.tensor.tensor.Tensor.__sub__", "syft.core.tensor.tensor.Tensor"),
        (
            "syft.core.tensor.tensor.Tensor.__truediv__",
            "syft.core.tensor.tensor.Tensor",
        ),
        ("syft.core.tensor.tensor.Tensor.argmax", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.argmin", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.argsort", "syft.core.tensor.tensor.Tensor"),
        # ("syft.core.tensor.tensor.Tensor.child", "syft.core.tensor.tensor.Tensor"),  # obj level
        ("syft.core.tensor.tensor.Tensor.clip", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.copy", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.cumprod", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.cumsum", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.diagonal", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.dot", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.flatten", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.max", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.mean", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.min", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.ndim", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.prod", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.repeat", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.reshape", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.resize", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.shape", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.sort", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.squeeze", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.std", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.sum", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.take", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.transpose", "syft.core.tensor.tensor.Tensor"),
        # SMPC
        (
            "syft.core.tensor.tensor.Tensor.fix_precision",
            "syft.core.tensor.tensor.Tensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        ("syft.core.tensor.tensor.Tensor.share", "syft.core.tensor.tensor.Tensor"),
        # Share Tensor Operations
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__add__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__sub__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__mul__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
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
