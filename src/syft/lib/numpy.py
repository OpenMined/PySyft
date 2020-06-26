import numpy as np
from ..ast import Globals

whitelist = set()
whitelist.add("numpy.array")
whitelist.add("numpy.ndarray")
whitelist.add("numpy.ndarray.__add__")

ast = Globals()

for method in whitelist:
    ast.add_path(method, np)

for klass in ast.classes:
    klass.create_pointer_class()
    klass.create_send_method()
