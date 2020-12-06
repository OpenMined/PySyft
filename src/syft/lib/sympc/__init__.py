# stdlib
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import sympc

# syft relative
from . import session  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals


def create_sympc_ast() -> Globals:
    ast = Globals()

    modules = [
        ("sympc"),
        ("sympc.session"),
        ("sympc.tensor"),
        ("sympc.protocol"),
        ("sympc.protocol.spdz"),
    ]

    classes = [
        ("sympc.session.Session", "sympc.session.Session", sympc.session.Session),
        # (
        #     "sympc.tensor.ShareTensor",
        #     "sympc.tensor.ShareTensor",
        #     sympc.tensor.ShareTensor,
        # ),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        # ("sympc.protocol.spdz.spdz.mul_parties", "sympc.tensor.ShareTensor"),
        # (
        #     "sympc.session.Session.przs_generate_random_share",
        #     "sympc.tensor.ShareTensor",
        # ),
        # (
        #     "sympc.session.get_generator",
        #     "torch.Generator",
        # ),
        # (
        #     "sympc.tensor.ShareTensor.__add__",
        #     "sympc.tensor.ShareTensor",
        # ),
        # (
        #     "sympc.tensor.ShareTensor.__sub__",
        #     "sympc.tensor.ShareTensor",
        # ),
        # (
        #     "sympc.tensor.ShareTensor.__mul__",
        #     "sympc.tensor.ShareTensor",
        # ),
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
