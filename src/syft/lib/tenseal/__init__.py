# stdlib
import sys
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

cpp_context = sys.modules["_tenseal_cpp"].TenSEALContext  # type: ignore


def create_tenseal_ast() -> Globals:
    ast = Globals()

    modules = ["tenseal", "tenseal._ts_cpp", "_tenseal_cpp"]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("tenseal.context", "tenseal.TenSEALContext", ts.context),
        (
            "_tenseal_cpp.TenSEALContext",
            "_tenseal_cpp.TenSEALContext",
            cpp_context,
        ),
        ("tenseal.SCHEME_TYPE", "tenseal.SCHEME_TYPE", ts.SCHEME_TYPE),
        (
            "tenseal._ts_cpp.CKKSVector",
            "tenseal._ts_cpp.CKKSVector",
            ts._ts_cpp.CKKSVector,
        ),
    ]

    methods = [
        ("tenseal.SCHEME_TYPE.BFV", "tenseal.SCHEME_TYPE"),
        ("tenseal.SCHEME_TYPE.CKKS", "tenseal.SCHEME_TYPE"),
        ("tenseal.SCHEME_TYPE.NONE", "tenseal.SCHEME_TYPE"),
        (
            "_tenseal_cpp.TenSEALContext.generate_galois_keys",
            "syft.lib.python._SyNone",
        ),
        (
            "_tenseal_cpp.TenSEALContext.global_scale",
            UnionGenerator["syft.lib.python.Int", "syft.lib.python.Float"],
        ),  # setter returns Int, getter returns Float?
        ("tenseal._ts_cpp.CKKSVector.__add__", "tenseal._ts_cpp.CKKSVector"),
        ("tenseal._ts_cpp.CKKSVector.dot", "tenseal._ts_cpp.CKKSVector"),
        ("tenseal._ts_cpp.CKKSVector.matmul", "tenseal._ts_cpp.CKKSVector"),
        ("tenseal._ts_cpp.CKKSVector.decrypt", "syft.lib.python.List"),
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
