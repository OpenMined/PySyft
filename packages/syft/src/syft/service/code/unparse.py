# stdlib
import _ast
import ast


def unparse(tree: _ast.Module) -> str:
    return ast.unparse(tree)
