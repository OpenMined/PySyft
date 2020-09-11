# import numpy as np
#
# # from .tensor import NumpyTensorWrapper
# # from .array import ArrayConstructor
# # from .ndarray import NdArrayConstructor
#
# # __all__ = ["NumpyTensorWrapper", "ArrayConstructor", "NdArrayConstructor"]
#
# from syft.ast.globals import Globals
#
# allowlist = set()
# allowlist.add("numpy.array")
# allowlist.add("numpy.ndarray")
# allowlist.add("numpy.ndarray.__add__")
#
#
# def create_numpy_ast():
#
#     ast = Globals()
#
#     for method in allowlist:
#         ast.add_path(path=method, framework_reference=np, return_type_name=None)
#
#     for klass in ast.classes:
#         # create syft classes
#         klass.create_pointer_class()
#         klass.create_send_method()
#         klass.create_serialization_methods()
#
#     return ast
