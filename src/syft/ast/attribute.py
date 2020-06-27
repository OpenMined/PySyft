from .. import ast
from abc import ABC

class Attribute(ABC):
    def __init__(self, name, path_and_name, ref):
        self.name = name  # __add__
        self.path_and_name = path_and_name  # torch.Tensor.__add__
        self.ref = ref  # <the actual add method object>
        self.attrs = {}  # any attrs of __add__ ... is none in this case

    @property
    def classes(self):
        out = list()

        if isinstance(self, ast.klass.Class):
            out.append(self)

        for name, ref in self.attrs.items():
            for klass in ref.classes:
                out.append(klass)
        return out

    @property
    def methods(self):
        out = list()

        if isinstance(self, ast.method.Method):
            out.append(self)

        for name, ref in self.attrs.items():
            for klass in ref.methods:
                out.append(klass)
        return out

    @property
    def functions(self):
        out = list()

        if isinstance(self, ast.function.Function):
            out.append(self)

        for name, ref in self.attrs.items():
            for klass in ref.functions:
                out.append(klass)
        return out

    @property
    def modules(self):
        out = list()

        if isinstance(self, ast.module.Module):
            out.append(self)

        for name, ref in self.attrs.items():
            for klass in ref.modules:
                out.append(klass)
        return out
