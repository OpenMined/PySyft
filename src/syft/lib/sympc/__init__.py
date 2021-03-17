# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# syft relative
from . import session  # noqa: 401
from . import share  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..util import generic_update_ast

LIB_NAME = "sympc"
PACKAGE_SUPPORT = {
    "lib": LIB_NAME,
    "torch": {"min_version": "1.6.0", "max_version": "1.8.0"},
    "python": {"min_version": (3, 7), "max_version": (3, 9, 99)},
}


def create_ast(client: TypeAny = None) -> Globals:
    # third party
    import sympc

    # syft relative
    from . import session  # noqa: 401
    from . import share  # noqa: 401

    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("sympc", sympc),
        ("sympc.session", sympc.session),
        ("sympc.tensor", sympc.tensor),
        ("sympc.protocol", sympc.protocol),
        ("sympc.store", sympc.store),
        ("sympc.protocol.fss", sympc.protocol.fss),
        ("sympc.protocol.fss.fss", sympc.protocol.fss.fss),
        ("sympc.protocol.spdz", sympc.protocol.spdz),
        ("sympc.protocol.spdz.spdz", sympc.protocol.spdz.spdz),
        ("sympc.utils", sympc.utils),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("sympc.session.Session", "sympc.session.Session", sympc.session.Session),
        ("sympc.store.CryptoStore", "sympc.store.CryptoStore", sympc.store.CryptoStore),
        (
            "sympc.tensor.ShareTensor",
            "sympc.tensor.ShareTensor",
            sympc.tensor.ShareTensor,
        ),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        ("sympc.store.CryptoStore.get_primitives_from_store", "syft.lib.python.List"),
        ("sympc.session.Session.crypto_store", "sympc.store.CryptoStore"),
        ("sympc.protocol.fss.fss.mask_builder", "sympc.tensor.ShareTensor"),
        ("sympc.protocol.fss.fss.evaluate", "sympc.tensor.ShareTensor"),
        ("sympc.protocol.spdz.spdz.mul_parties", "sympc.tensor.ShareTensor"),
        ("sympc.protocol.spdz.spdz.spdz_mask", "syft.lib.python.Tuple"),
        ("sympc.protocol.spdz.spdz.div_wraps", "sympc.tensor.ShareTensor"),
        (
            "sympc.session.Session.przs_generate_random_share",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.store.CryptoStore.populate_store",
            "syft.lib.python._SyNone",
        ),
        (
            "sympc.utils.get_new_generator",
            "torch.Generator",
        ),
        (
            "sympc.tensor.ShareTensor.__add__",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.tensor.ShareTensor.__sub__",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.tensor.ShareTensor.__rmul__",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.tensor.ShareTensor.__mul__",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.tensor.ShareTensor.__matmul__",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.tensor.ShareTensor.__truediv__",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.tensor.ShareTensor.__rmatmul__",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.tensor.ShareTensor.numel",
            "syft.lib.python.Int",  # FIXME: Can't we just return an int??
        ),
        (
            "sympc.tensor.ShareTensor.T",
            "sympc.tensor.ShareTensor",
        ),
        ("sympc.tensor.ShareTensor.unsqueeze", "sympc.tensor.ShareTensor"),
        ("sympc.tensor.ShareTensor.view", "sympc.tensor.ShareTensor"),
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
