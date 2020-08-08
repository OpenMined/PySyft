import numpy as np
from .tensor import NumpyTensorWrapper
from .array import ArrayConstructor
from .ndarray import NdArrayConstructor

__all__ = ["NumpyTensorWrapper", "ArrayConstructor", "NdArrayConstructor"]

from syft.ast.globals import Globals

whitelist = set()
whitelist.add("numpy.array")
whitelist.add("numpy.ndarray")
whitelist.add("numpy.ndarray.__add__")


def create_numpy_ast():

    ast = Globals()

    for method in whitelist:
        ast.add_path(path=method, framework_reference=np, return_type_name=None)

    for klass in ast.classes:
        # create syft classes
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()

    return ast
