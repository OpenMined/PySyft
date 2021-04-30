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
        ("numpy.flagsobj", "numpy.flagsobj", np.flagsobj),
        ("numpy.flatiter", "numpy.flatiter", np.flatiter),
        ("numpy.core._internal._ctypes", "numpy.core._internal._ctypes", np.core._internal._ctypes),
        ("numpy.flatiter", "numpy.flatiter", np.flatiter),
    ]

    methods: TypeList[TypeTuple[str, str]] = [
        # Array attributes
        ("numpy.ndarray.flags", "numpy.flagsobj"), 
        ("numpy.ndarray.shape", "syft.lib.python.Tuple"),
        ("numpy.ndarray.strides", "syft.lib.python.Tuple"),
        ("numpy.ndarray.ndim", "syft.lib.python.Int"),
        ("numpy.ndarray.data", "syft.lib.python.memoryview"), # serde Req
        ("numpy.ndarray.size", "syft.lib.python.Int"),
        ("numpy.ndarray.itemsize", "syft.lib.python.Int"),
        ("numpy.ndarray.nbytes", "syft.lib.python.Int"),
        ("numpy.ndarray.base", "syft.lib.python._SyNone"),
        # ("numpy.ndarray.dtype", "numpy.ndarray"), # SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET
        ("numpy.ndarray.T", "numpy.ndarray"),
        # ("numpy.ndarray.real", "numpy.ndarray"), # requires dtype complex
        # ("numpy.ndarray.imag", "numpy.ndarray"), # requires dtype complex
        ("numpy.ndarray.flat", "numpy.flatiter"), 
        ("numpy.ndarray.ctypes", "numpy.core._internal._ctypes"), 
        ("numpy.ndarray.__array_interface__", "syft.lib.python.Dict"),
        ("numpy.ndarray.__array_struct__", "syft.lib.python.PyCapsule"), # serde Req
        # Array methods - Array Conversion
        ("numpy.ndarray.item", "syft.lib.python.Int"), 
        ("numpy.ndarray.tolist", "syft.lib.python.List"), # Ask if required 
        # ("numpy.ndarray.dump", "syft.lib.python.builtin_function_or_method"), 
        # ("numpy.ndarray.dumps", "syft.lib.python.builtin_function_or_method"),
        ("numpy.ndarray.astype", "numpy.ndarray"), # Ask if required 
        ("numpy.ndarray.byteswap", "numpy.ndarray"),
        ("numpy.ndarray.copy", "numpy.ndarray"),
        ("numpy.ndarray.view", "numpy.ndarray"),
        ("numpy.ndarray.getfield", "numpy.ndarray"),
        ("numpy.ndarray.setflags", "numpy.ndarray"),
        ("numpy.ndarray.fill", "numpy.ndarray"),
        # Do above Tests
        # Array methods - Shape Manipulation
        ("numpy.ndarray.reshape", "numpy.ndarray"),
        ("numpy.ndarray.resize", "numpy.ndarray"),
        ("numpy.ndarray.transpose", "numpy.ndarray"),
        ("numpy.ndarray.swapaxes", "numpy.ndarray"),
        ("numpy.ndarray.flatten", "numpy.ndarray"),
        ("numpy.ndarray.ravel", "numpy.ndarray"),
        ("numpy.ndarray.squeeze", "numpy.ndarray"),
        # Array methods - Item Selection and Manipulation
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
        # Array methods - Calculation
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
        # Comparison operators
        ("numpy.ndarray.__lt__", "numpy.ndarray"),
        ("numpy.ndarray.__le__", "numpy.ndarray"),
        ("numpy.ndarray.__gt__", "numpy.ndarray"),
        ("numpy.ndarray.__ge__", "numpy.ndarray"),
        ("numpy.ndarray.__eq__", "numpy.ndarray"),
        ("numpy.ndarray.__ne__", "numpy.ndarray"),
        # Truth value of an array
        ("numpy.ndarray.__bool__", "numpy.ndarray"),
        # Unary operations
        ("numpy.ndarray.__neg__", "numpy.ndarray"),
        ("numpy.ndarray.__pos__", "numpy.ndarray"),
        ("numpy.ndarray.__abs__", "numpy.ndarray"),
        ("numpy.ndarray.__invert__", "numpy.ndarray"),
        # Arithmetic
        ("numpy.ndarray.__add__", "numpy.ndarray"),
        ("numpy.ndarray.__sub__", "numpy.ndarray"),
        ("numpy.ndarray.__mul__", "numpy.ndarray"),
        ("numpy.ndarray.__truediv__", "numpy.ndarray"),
        ("numpy.ndarray.__floordiv__", "numpy.ndarray"),
        ("numpy.ndarray.__mod__", "numpy.ndarray"),
        ("numpy.ndarray.__divmod__", "numpy.ndarray"),
        ("numpy.ndarray.__pow__", "numpy.ndarray"),
        ("numpy.ndarray.__lshift__", "numpy.ndarray"),
        ("numpy.ndarray.__rshift__", "numpy.ndarray"),
        ("numpy.ndarray.__and__", "numpy.ndarray"),
        ("numpy.ndarray.__or__", "numpy.ndarray"),
        ("numpy.ndarray.__xor__", "numpy.ndarray"),
        # Arithmetic, in-place
        ("numpy.ndarray.__iadd__", "numpy.ndarray"),
        ("numpy.ndarray.__isub__", "numpy.ndarray"),
        ("numpy.ndarray.__imul__", "numpy.ndarray"),
        ("numpy.ndarray.__itruediv__", "numpy.ndarray"),
        ("numpy.ndarray.__ifloordiv__", "numpy.ndarray"),
        ("numpy.ndarray.__imod__", "numpy.ndarray"),
        ("numpy.ndarray.__ipow__", "numpy.ndarray"),
        ("numpy.ndarray.__ilshift__", "numpy.ndarray"),
        ("numpy.ndarray.__irshift__", "numpy.ndarray"),
        ("numpy.ndarray.__iand__", "numpy.ndarray"),
        ("numpy.ndarray.__ior__", "numpy.ndarray"),
        ("numpy.ndarray.__ixor__", "numpy.ndarray"),
        # Matrix Multiplication
        ("numpy.ndarray.__matmul__", "numpy.ndarray"),
        # Special Methods
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
