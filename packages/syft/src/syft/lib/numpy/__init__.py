# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import numpy as np

# relative
from . import array  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..misc.union import UnionGenerator
from ..util import generic_update_ast

LIB_NAME = "numpy"
PACKAGE_SUPPORT = {"lib": LIB_NAME}


def create_ast(client: TypeAny = None) -> Globals:
    ast = Globals(client)

    modules: TypeList[TypeTuple[str, TypeAny]] = [
        ("numpy", np),
    ]

    classes: TypeList[TypeTuple[str, str, TypeAny]] = [
        ("numpy.ndarray", "numpy.ndarray", np.ndarray),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        ("numpy.ndarray.shape", "syft.lib.python.Tuple"),
        ("numpy.ndarray.strides", "syft.lib.python.Tuple"),
        ("numpy.ndarray.ndim", "syft.lib.python.Int"),
        ("numpy.ndarray.size", "syft.lib.python.Int"),
        ("numpy.ndarray.itemsize", "syft.lib.python.Int"),
        ("numpy.ndarray.nbytes", "syft.lib.python.Int"),
        # ("numpy.ndarray.dtype", "numpy.ndarray"), # SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET
        ("numpy.ndarray.T", "numpy.ndarray"),
        # ("numpy.ndarray.real", "numpy.ndarray"), # requires dtype complex
        # ("numpy.ndarray.imag", "numpy.ndarray"), # requires dtype complex
        # Array methods - Array Conversion
        (
            "numpy.ndarray.item",
            UnionGenerator[
                "syft.lib.python.Bool", "syft.lib.python.Float", "syft.lib.python.Int"
            ],
        ),
        ("numpy.ndarray.byteswap", "numpy.ndarray"),
        ("numpy.ndarray.copy", "numpy.ndarray"),
        ("numpy.ndarray.view", "numpy.ndarray"),
        ("numpy.ndarray.__add__", "numpy.ndarray"),
        ("numpy.ndarray.sum", "numpy.ndarray"),
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
