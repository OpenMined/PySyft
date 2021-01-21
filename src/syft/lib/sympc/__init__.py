# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple
from typing import Union as TypeUnion

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals

PACKAGE_SUPPORT = {"lib": "sympc", "torch": {"min_version": "1.6.0"}}


# this gets called on global ast as well as clients
# anything which wants to have its ast updated and has an add_attr method
def update_ast(ast: TypeUnion[Globals, TypeAny], client: TypeAny = None) -> None:
    sympc_ast = create_ast(client=client)
    ast.add_attr(attr_name="sympc", attr=sympc_ast.attrs["sympc"])


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
        ("sympc.protocol.spdz", sympc.protocol.spdz),
        ("sympc.protocol.spdz.spdz", sympc.protocol.spdz.spdz),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("sympc.session.Session", "sympc.session.Session", sympc.session.Session),
        (
            "sympc.tensor.ShareTensor",
            "sympc.tensor.ShareTensor",
            sympc.tensor.ShareTensor,
        ),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        ("sympc.protocol.spdz.spdz.mul_parties", "sympc.tensor.ShareTensor"),
        (
            "sympc.session.Session.przs_generate_random_share",
            "sympc.tensor.ShareTensor",
        ),
        (
            "sympc.session.get_generator",
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
            "sympc.tensor.ShareTensor.__rmatmul__",
            "sympc.tensor.ShareTensor",
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
