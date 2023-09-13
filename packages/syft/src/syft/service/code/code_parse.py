# stdlib
import ast


class GlobalsVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        if isinstance(node, ast.Global):
            raise Exception("No Globals allowed!")
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Call(self, node):
        print(node.func)
        if isinstance(node.func, ast.Name):
            logging_attr = ast.Attribute(value=ast.Name(id='logger'), attr='print', ctx=ast.Load())
            node.func = logging_attr
        self.generic_visit(node)