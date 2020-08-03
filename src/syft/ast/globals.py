# from .. import ast # CAUSES Circular import errors
from .module import Module
from .util import unsplit


class Globals(Module):

    """The collection of frameworks held in a global namespace"""

    def __init__(self):
        super().__init__("globals", None, None, None)

    def add_framework(self, attr_name, attr):
        self.attrs[attr_name] = attr

    def add_path(self, path, return_type_name=None, framework_reference=None):
        if isinstance(path, str):
            path = path.split(".")

        framework_name = path[0]

        if framework_name not in self.attrs:
            if framework_reference is not None:
                self.attrs[framework_name] = Module(
                    name=framework_name,
                    path_and_name=unsplit(path),
                    ref=framework_reference,
                    return_type_name=return_type_name,
                )
            else:
                raise Exception(
                    "You must pass in a framework object the first time you add method "
                    "within a framework."
                )

        self.attrs[framework_name].add_path(path, 1, return_type_name=return_type_name)
