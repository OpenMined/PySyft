# stdlib
import ast


class GlobalsVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        if isinstance(node, ast.Global):
            raise Exception("No Globals allowed!")
        ast.NodeVisitor.generic_visit(self, node)
