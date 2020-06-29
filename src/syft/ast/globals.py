from .. import ast

from .util import unsplit


class Globals(ast.module.Module):

    """The collection of frameworks held in a global namespace"""

    def __init__(self):
        super().__init__("globals", None, None)

    def add_framework(self, attr_name, attr):
        self.attrs[attr_name] = attr

    def add_path(self, path, framework_reference=None):
        if isinstance(path, str):
            path = path.split(".")

        framework_name = path[0]

        if framework_name not in self.attrs:
            if framework_reference is not None:
                self.attrs[framework_name] = ast.module.Module(
                    framework_name, unsplit(path), framework_reference
                )
            else:
                raise Exception(
                    "You must pass in a framework object the first time you add method within a framework."
                )

        self.attrs[framework_name].add_path(path, 1)
