from . import torch
from . import numpy
from ..ast.globals import Globals

supported_frameworks = [torch, numpy]

# now we need to load the relevant frameworks onto the node
lib_ast = Globals()
lib_ast.add_attr(attr_name="torch", attr=torch.ast.attrs['torch'])
lib_ast.add_attr(attr_name="numpy", attr=numpy.ast.attrs['numpy'])