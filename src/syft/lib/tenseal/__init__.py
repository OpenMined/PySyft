# stdlib
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

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


def create_tenseal_ast() -> Globals:
    ast = Globals()

    modules = ["tenseal"]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("tenseal.Context", "tenseal.Context", ts.Context),
        ("tenseal.SCHEME_TYPE", "tenseal.SCHEME_TYPE", ts.SCHEME_TYPE),
        ("tenseal.CKKSVector", "tenseal.CKKSVector", ts.CKKSVector),
    ]

    methods = [
        ("tenseal.SCHEME_TYPE.BFV", "tenseal.SCHEME_TYPE"),
        ("tenseal.SCHEME_TYPE.CKKS", "tenseal.SCHEME_TYPE"),
        ("tenseal.SCHEME_TYPE.NONE", "tenseal.SCHEME_TYPE"),
        ## Context
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
        ),  # setter returns Int, getter returns Float?
        ("tenseal.Context.auto_mod_switch", "syft.lib.python.Bool"),
        ("tenseal.Context.auto_relin", "syft.lib.python.Bool"),
        ("tenseal.Context.auto_rescale", "syft.lib.python.Bool"),
        ## CKKSVector
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
