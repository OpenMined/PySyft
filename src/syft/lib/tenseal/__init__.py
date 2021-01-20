# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple
from typing import Union as TypeUnion

# third party
import tenseal as ts

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..misc.union import UnionGenerator
from .ckks_vector import CKKSVector  # noqa: 401
from .context import ContextWrapper  # noqa: 401

LIB_NAME = "tenseal"
PACKAGE_SUPPORT = {"lib": LIB_NAME}


# this gets called on global ast as well as clients
# anything which wants to have its ast updated and has an add_attr method
def update_ast(ast: TypeUnion[Globals, TypeAny], client: TypeAny = None) -> None:
    tenseal_ast = create_ast(client=client)
    ast.add_attr(attr_name=LIB_NAME, attr=tenseal_ast.attrs[LIB_NAME])


def create_ast(client: TypeAny) -> Globals:
    ast = Globals(client=client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [("tenseal", ts)]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("tenseal.Context", "tenseal.Context", ts.Context),
        ("tenseal.SCHEME_TYPE", "tenseal.SCHEME_TYPE", ts.SCHEME_TYPE),
        ("tenseal.CKKSVector", "tenseal.CKKSVector", ts.CKKSVector),
    ]

    methods = [
        ("tenseal.SCHEME_TYPE.BFV", "tenseal.SCHEME_TYPE"),
        ("tenseal.SCHEME_TYPE.CKKS", "tenseal.SCHEME_TYPE"),
        ("tenseal.SCHEME_TYPE.NONE", "tenseal.SCHEME_TYPE"),
        # Context
        ("tenseal.Context.generate_galois_keys", "syft.lib.python._SyNone"),
        ("tenseal.Context.generate_relin_keys", "syft.lib.python._SyNone"),
        ("tenseal.Context.has_galois_keys", "syft.lib.python.Bool"),
        ("tenseal.Context.has_relin_keys", "syft.lib.python.Bool"),
        ("tenseal.Context.has_public_key", "syft.lib.python.Bool"),
        ("tenseal.Context.has_secret_key", "syft.lib.python.Bool"),
        ("tenseal.Context.is_public", "syft.lib.python.Bool"),
        ("tenseal.Context.is_private", "syft.lib.python.Bool"),
        ("tenseal.Context.make_context_public", "syft.lib.python._SyNone"),
        ("tenseal.Context.copy", "tenseal.Context"),
        # Attr and property doesn't currently work
        (
            "tenseal.Context.global_scale",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),
        ("tenseal.Context.auto_mod_switch", "syft.lib.python.Bool"),
        ("tenseal.Context.auto_relin", "syft.lib.python.Bool"),
        ("tenseal.Context.auto_rescale", "syft.lib.python.Bool"),
        # CKKSVector
        ("tenseal.CKKSVector.__add__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.dot", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.matmul", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.decrypt", "syft.lib.python.List"),
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
