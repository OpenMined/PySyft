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
        ("numpy.ndarray.flags", "numpy.ndarray.flags"),
        ("numpy.ndarray.shape", "numpy.ndarray.shape"),
        ("numpy.ndarray.strides", "numpy.ndarray.strides"),
        ("numpy.ndarray.ndim", "numpy.ndarray.ndim"),
        ("numpy.ndarray.data", "numpy.ndarray.data"),
        ("numpy.ndarray.size", "numpy.ndarray.size"),
        ("numpy.ndarray.itemsize", "numpy.ndarray.itemsize"),
        ("numpy.ndarray.nbytes", "numpy.ndarray.nbytes"),
        ("numpy.ndarray.base", "numpy.ndarray.base"),
        ("numpy.ndarray.dtype", "numpy.ndarray.dtype"),
        ("numpy.ndarray.T", "numpy.ndarray.T"),
        ("numpy.ndarray.real", "numpy.ndarray.real"),
        ("numpy.ndarray.imag", "numpy.ndarray.imag"),
        ("numpy.ndarray.flat", "numpy.ndarray.flat"),
        ("numpy.ndarray.ctypes", "numpy.ndarray.ctypes"),
        ("numpy.ndarray.__array_interface__", "numpy.ndarray.__array_interface__"),
        ("numpy.ndarray.__array_struct__", "numpy.ndarray.__array_struct__"),
        ("numpy.ndarray.ctypes", "numpy.ndarray.ctypes"),
        # Array methods
        ("numpy.ndarray.item", "numpy.ndarray.item"),
        ("numpy.ndarray.tolist", "numpy.ndarray.tolist"),
        ("numpy.ndarray.itemset", "numpy.ndarray.itemset"),
        ("numpy.ndarray.tostring", "numpy.ndarray.tostring"),
        ("numpy.ndarray.tobytes", "numpy.ndarray.tobytes"),
        ("numpy.ndarray.tofile", "numpy.ndarray.tofile"),
        ("numpy.ndarray.dump", "numpy.ndarray.dump"),
        ("numpy.ndarray.dumps", "numpy.ndarray.dumps"),
        ("numpy.ndarray.astype", "numpy.ndarray.astype"),
        ("numpy.ndarray.byteswap", "numpy.ndarray.byteswap"),
        ("numpy.ndarray.copy", "numpy.ndarray.copy"),
        ("numpy.ndarray.view", "numpy.ndarray.view"),
        ("numpy.ndarray.getfield", "numpy.ndarray.getfield"),
        ("numpy.ndarray.setflags", "numpy.ndarray.setflags"),
        ("numpy.ndarray.fill", "numpy.ndarray.fill"),
        ("numpy.ndarray.reshape", "numpy.ndarray.reshape"),
        ("numpy.ndarray.resize", "numpy.ndarray.resize"),
        ("numpy.ndarray.transpose", "numpy.ndarray.transpose"),
        ("numpy.ndarray.swapaxes", "numpy.ndarray.swapaxes"),
        ("numpy.ndarray.flatten", "numpy.ndarray.flatten"),
        ("numpy.ndarray.ravel", "numpy.ndarray.ravel"),
        ("numpy.ndarray.squeeze", "numpy.ndarray.squeeze"),
        ("numpy.ndarray.take", "numpy.ndarray.take"),
        ("numpy.ndarray.put", "numpy.ndarray.put"),
        ("numpy.ndarray.repeat", "numpy.ndarray.repeat"),
        ("numpy.ndarray.choose", "numpy.ndarray.choose"),
        ("numpy.ndarray.sort", "numpy.ndarray.sort"),
        ("numpy.ndarray.argsort", "numpy.ndarray.argsort"),
        ("numpy.ndarray.partition", "numpy.ndarray.partition"),
        ("numpy.ndarray.argpartition", "numpy.ndarray.argpartition"),
        ("numpy.ndarray.searchsorted", "numpy.ndarray.searchsorted"),
        ("numpy.ndarray.nonzero", "numpy.ndarray.nonzero"),
        ("numpy.ndarray.compress", "numpy.ndarray.compress"),
        ("numpy.ndarray.diagonal", "numpy.ndarray.diagonal"),
        ("numpy.ndarray.max", "numpy.ndarray.max"),
        ("numpy.ndarray.argmax", "numpy.ndarray.argmax"),
        ("numpy.ndarray.min", "numpy.ndarray.min"),
        ("numpy.ndarray.argmin", "numpy.ndarray.argmin"),
        ("numpy.ndarray.ptp", "numpy.ndarray.ptp"),
        ("numpy.ndarray.clip", "numpy.ndarray.clip"),
        ("numpy.ndarray.conj", "numpy.ndarray.conj"),
        ("numpy.ndarray.round", "numpy.ndarray.round"),
        ("numpy.ndarray.trace", "numpy.ndarray.trace"),
        ("numpy.ndarray.sum", "numpy.ndarray.sum"),
        ("numpy.ndarray.cumsum", "numpy.ndarray.cumsum"),
        ("numpy.ndarray.mean", "numpy.ndarray.mean"),
        ("numpy.ndarray.var", "numpy.ndarray.var"),
        ("numpy.ndarray.std", "numpy.ndarray.std"),
        ("numpy.ndarray.prod", "numpy.ndarray.prod"),
        ("numpy.ndarray.cumprod", "numpy.ndarray.cumprod"),
        ("numpy.ndarray.all", "numpy.ndarray.all"),
        ("numpy.ndarray.any", "numpy.ndarray.any"),
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
