# stdlib
from _ast import Module
import ast
from typing import Any


class GlobalsVisitor(ast.NodeVisitor):
    def generic_visit(self, node: Any) -> None:
        if isinstance(node, ast.Global):
            raise Exception("No Globals allowed!")
        ast.NodeVisitor.generic_visit(self, node)


class LaunchJobVisitor(ast.NodeVisitor):
    def visit_Module(self, node: Module) -> Any:
        self.nested_calls: list = []
        self.generic_visit(node)

    def visit_Call(self, node: Any) -> None:
        if isinstance(node.func, ast.Attribute):
            if (
                getattr(node.func.value, "id", None) == "datasite"
                and node.func.attr == "launch_job"
            ):
                self.nested_calls.append(node.args[0].id)
