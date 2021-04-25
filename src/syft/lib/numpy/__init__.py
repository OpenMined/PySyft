# stdlib
import functools
from typing import Any as TypeAny
from typing import List as TypeList
from typing import Tuple as TypeTuple

# third party
import numpy as np

# syft relative
from . import array  # noqa: 401
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
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
        # Array attributes
        ("numpy.ndarray.flags", "numpy.ndarray"),
        ("numpy.ndarray.shape", "numpy.ndarray"),
        ("numpy.ndarray.strides", "numpy.ndarray"),
        ("numpy.ndarray.ndim", "numpy.ndarray"),
        ("numpy.ndarray.data", "numpy.ndarray"),
        ("numpy.ndarray.size", "numpy.ndarray"),
        ("numpy.ndarray.itemsize", "numpy.ndarray"),
        ("numpy.ndarray.nbytes", "numpy.ndarray"),
        ("numpy.ndarray.base", "numpy.ndarray"),
        # ("numpy.ndarray.dtype", "numpy.ndarray"), # SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET
        ("numpy.ndarray.T", "numpy.ndarray"),
        # ("numpy.ndarray.real", "numpy.ndarray"), # requires dtype complex
        # ("numpy.ndarray.imag", "numpy.ndarray"), # requires dtype complex
        ("numpy.ndarray.flat", "numpy.ndarray"),
        ("numpy.ndarray.ctypes", "numpy.ndarray"),
        ("numpy.ndarray.__array_interface__", "numpy.ndarray"),
        ("numpy.ndarray.__array_struct__", "numpy.ndarray"),
        # Array methods
        ("numpy.ndarray.item", "numpy.ndarray"),
        ("numpy.ndarray.tolist", "numpy.ndarray"),
        ("numpy.ndarray.itemset", "numpy.ndarray"),
        ("numpy.ndarray.tostring", "numpy.ndarray"),
        ("numpy.ndarray.tobytes", "numpy.ndarray"),
        ("numpy.ndarray.tofile", "numpy.ndarray"),
        ("numpy.ndarray.dump", "numpy.ndarray"),
        ("numpy.ndarray.dumps", "numpy.ndarray"),
        ("numpy.ndarray.astype", "numpy.ndarray"),
        ("numpy.ndarray.byteswap", "numpy.ndarray"),
        ("numpy.ndarray.copy", "numpy.ndarray"),
        ("numpy.ndarray.view", "numpy.ndarray"),
        ("numpy.ndarray.getfield", "numpy.ndarray"),
        ("numpy.ndarray.setflags", "numpy.ndarray"),
        ("numpy.ndarray.fill", "numpy.ndarray"),
        ("numpy.ndarray.reshape", "numpy.ndarray"),
        ("numpy.ndarray.resize", "numpy.ndarray"),
        ("numpy.ndarray.transpose", "numpy.ndarray"),
        ("numpy.ndarray.swapaxes", "numpy.ndarray"),
        ("numpy.ndarray.flatten", "numpy.ndarray"),
        ("numpy.ndarray.ravel", "numpy.ndarray"),
        ("numpy.ndarray.squeeze", "numpy.ndarray"),
        ("numpy.ndarray.take", "numpy.ndarray"),
        ("numpy.ndarray.put", "numpy.ndarray"),
        ("numpy.ndarray.repeat", "numpy.ndarray"),
        ("numpy.ndarray.choose", "numpy.ndarray"),
        ("numpy.ndarray.sort", "numpy.ndarray"),
        ("numpy.ndarray.argsort", "numpy.ndarray"),
        ("numpy.ndarray.partition", "numpy.ndarray"),
        ("numpy.ndarray.argpartition", "numpy.ndarray"),
        ("numpy.ndarray.searchsorted", "numpy.ndarray"),
        ("numpy.ndarray.nonzero", "numpy.ndarray"),
        ("numpy.ndarray.compress", "numpy.ndarray"),
        ("numpy.ndarray.diagonal", "numpy.ndarray"),
        ("numpy.ndarray.max", "numpy.ndarray"),
        ("numpy.ndarray.argmax", "numpy.ndarray"),
        ("numpy.ndarray.min", "numpy.ndarray"),
        ("numpy.ndarray.argmin", "numpy.ndarray"),
        ("numpy.ndarray.ptp", "numpy.ndarray"),
        ("numpy.ndarray.clip", "numpy.ndarray"),
        ("numpy.ndarray.conj", "numpy.ndarray"),
        ("numpy.ndarray.round", "numpy.ndarray"),
        ("numpy.ndarray.trace", "numpy.ndarray"),
        ("numpy.ndarray.sum", "numpy.ndarray"),
        ("numpy.ndarray.cumsum", "numpy.ndarray"),
        ("numpy.ndarray.mean", "numpy.ndarray"),
        ("numpy.ndarray.var", "numpy.ndarray"),
        ("numpy.ndarray.std", "numpy.ndarray"),
        ("numpy.ndarray.prod", "numpy.ndarray"),
        ("numpy.ndarray.cumprod", "numpy.ndarray"),
        ("numpy.ndarray.all", "numpy.ndarray"),
        ("numpy.ndarray.any", "numpy.ndarray"),
        # Arithmetic, matrix multiplication, and comparison operations
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
