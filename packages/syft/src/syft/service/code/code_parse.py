# stdlib
from _ast import Module
import ast
from typing import Any


class GlobalsVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        if isinstance(node, ast.Global):
            raise Exception("No Globals allowed!")
        ast.NodeVisitor.generic_visit(self, node)

nested_calls = []

def reset_nested_calls():
    global nested_calls
    nested_calls = []

def get_nested_calls():
    return nested_calls

class LaunchJobVisitor(ast.NodeVisitor):
    def visit_Module(self, node: Module) -> Any:
        self.nested_calls = []
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
                if node.func.value.id == "domain" and node.func.attr == "launch_job": 
                    self.nested_calls.append(node.args[0].id)
    
    # def generic_visit(self, node):
    #     if isinstance(node, ast.Call):
    #         if isinstance(node.func, ast.Attribute):
    #             if node.func.value.id == "domain" and node.func.attr == "launch_job": 
    #                 nested_calls.append(node.args[0].id)
    #     ast.NodeVisitor.generic_visit(self, node)