# stdlib
from typing import Optional

# relative
from . import functions  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..node.abstract.node import AbstractNodeClient
from .autodp.gamma_tensor import GammaTensor
from .autodp.phi_tensor import PhiTensor
from .fixed_precision_tensor import FixedPrecisionTensor
from .nn import Model
from .smpc.share_tensor import ShareTensor
from .tensor import Tensor
from .tensor import TensorPointer  # noqa: 401


def create_tensor_ast(client: Optional[AbstractNodeClient] = None) -> Globals:
    ast = Globals(client)

    modules = [
        "syft",
        "syft.core",
        "syft.core.tensor",
        "syft.core.tensor.tensor",
        "syft.core.tensor.smpc",
        "syft.core.tensor.smpc.share_tensor",
        "syft.core.tensor.fixed_precision_tensor",
        "syft.core.tensor.autodp",
        "syft.core.tensor.autodp.phi_tensor",
        "syft.core.tensor.autodp.gamma_tensor",
        "syft.core.tensor.nn",
    ]
    classes = [
        ("syft.core.tensor.tensor.Tensor", "syft.core.tensor.tensor.Tensor", Tensor),
        (
            "syft.core.tensor.autodp.phi_tensor.PhiTensor",
            "syft.core.tensor.autodp.phi_tensor.PhiTensor",
            PhiTensor,
        ),
        (
            "syft.core.tensor.autodp.gamma_tensor.GammaTensor",
            "syft.core.tensor.autodp.gamma_tensor.GammaTensor",
            GammaTensor,
        ),
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
        (
            "syft.core.tensor.nn.Model",
            "syft.core.tensor.nn.Model",
            Model,
        ),
    ]

    methods = [
        # # Tensor
        ("syft.core.tensor.tensor.Tensor.T", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__abs__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__add__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__and__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__or__", "syft.core.tensor.tensor.Tensor"),
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
        ("syft.core.tensor.tensor.Tensor.__xor__", "syft.core.tensor.tensor.Tensor"),
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
        ("syft.core.tensor.tensor.Tensor.__round__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.round", "syft.core.tensor.tensor.Tensor"),
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
        # # ("syft.core.tensor.tensor.Tensor.backward", "syft.lib.python.Bool"),
        # # ("syft.core.tensor.tensor.Tensor.child", "syft.core.tensor.tensor.Tensor"),  # obj level
        ("syft.core.tensor.tensor.Tensor.choose", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.clip", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.copy", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.cumprod", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.cumsum", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.diagonal", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.dot", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.flatten", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.ravel", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.compress", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.swapaxes", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.gamma", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.max", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.mean", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.min", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.ndim", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.private", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.prod", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.repeat", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.reshape", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.resize", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__mod__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.shape", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.sort", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.squeeze", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.std", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.var", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.sum", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.take", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.tag", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.transpose", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.__pos__", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.put", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.trace", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.ptp", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.all", "syft.core.tensor.tensor.Tensor"),
        ("syft.core.tensor.tensor.Tensor.any", "syft.core.tensor.tensor.Tensor"),
        (
            "syft.core.tensor.tensor.Tensor.bit_decomposition",
            "syft.lib.python._SyNone",
        ),
        (
            "syft.core.tensor.tensor.Tensor.concatenate",
            "syft.core.tensor.tensor.Tensor",
        ),
        (
            "syft.core.tensor.tensor.Tensor.ones_like",
            "syft.core.tensor.tensor.Tensor",
        ),
        # # SMPC
        # (
        #     "syft.core.tensor.tensor.Tensor.fix_precision",
        #     "syft.core.tensor.tensor.Tensor",
        # ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs",
            "syft.core.tensor.tensor.Tensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs_on_dp_tensor",
            "syft.core.tensor.tensor.Tensor",
        ),
        # ("syft.core.tensor.tensor.Tensor.share", "syft.core.tensor.tensor.Tensor"),
        # Share Tensor Operations
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__add__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__and__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__or__",
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
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__gt__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__matmul__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__rmatmul__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.sum",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.repeat",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.copy",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.diagonal",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.flatten",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.transpose",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.resize",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.ravel",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.compress",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.reshape",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.squeeze",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.swapaxes",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__pos__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.put",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.__neg__",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.take",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.cumsum",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.trace",
            "syft.core.tensor.smpc.share_tensor.ShareTensor",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.crypto_store",
            "syft.core.smpc.store.CryptoStore",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.populate_store",
            "syft.lib.python._SyNone",
        ),
        (
            "syft.core.tensor.smpc.share_tensor.ShareTensor.bit_decomposition",
            "syft.lib.python._SyNone",
        ),
        # nn Modules
        (
            "syft.core.tensor.nn.Model.fit",
            "syft.lib.python._SyNone",
        ),
        (
            "syft.core.tensor.nn.Model.step",
            "syft.lib.python._SyNone",
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
