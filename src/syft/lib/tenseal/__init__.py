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
from .context import ContextWrapper  # noqa: 401

from .ckks_vector import CKKSVector  # noqa: 401
from .ckks_tensor import CKKSTensor  # noqa: 401
from .bfv_vector import BFVVector  # noqa: 401

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
        ("tenseal.CKKSTensor", "tenseal.CKKSTensor", ts.CKKSTensor),
        ("tenseal.BFVVector", "tenseal.BFVVector", ts.BFVVector),
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
        ("tenseal.CKKSVector.__iadd__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__radd__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__mul__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__imul__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__rmul__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__sub__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__isub__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__rsub__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__pow__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__ipow__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__neg__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.add", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.add_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.mul", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.mul_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.sub", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.sub_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.neg", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.neg_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.sum", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.sum_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.square", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.square_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.pow", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.pow_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.polyval", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.polyval_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.dot", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector._dot", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.dot_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.mm", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector._mm", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.mm_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.matmul", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.matmul_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__matmul__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.__imatmul__", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.conv2d_im2col", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.conv2d_im2col_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector._enc_matmul_plain", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.enc_matmul_plain", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.enc_matmul_plain_", "tenseal.CKKSVector"),
        ("tenseal.CKKSVector.decrypt", "syft.lib.python.List"),
        ("tenseal.CKKSVector.link_context", "syft.lib.python._SyNone"),
        # BFVVector
        ("tenseal.BFVVector.__add__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.__iadd__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.__radd__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.__mul__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.__imul__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.__rmul__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.__sub__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.__isub__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.__rsub__", "tenseal.BFVVector"),
        ("tenseal.BFVVector.add", "tenseal.BFVVector"),
        ("tenseal.BFVVector.add_", "tenseal.BFVVector"),
        ("tenseal.BFVVector.mul", "tenseal.BFVVector"),
        ("tenseal.BFVVector.mul_", "tenseal.BFVVector"),
        ("tenseal.BFVVector.sub", "tenseal.BFVVector"),
        ("tenseal.BFVVector.sub_", "tenseal.BFVVector"),
        ("tenseal.BFVVector.decrypt", "syft.lib.python.List"),
        ("tenseal.BFVVector.link_context", "syft.lib.python._SyNone"),
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
