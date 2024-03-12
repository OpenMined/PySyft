# stdlib
import ast
from typing import Any

# relative
from .unparse import unparse


class GlobalsVisitor(ast.NodeVisitor):
    def generic_visit(self, node: Any) -> None:
        if isinstance(node, ast.Global):
            raise Exception("No Globals allows")
        ast.NodeVisitor.generic_visit(self, node)


def check_no_returns(module: ast.Module) -> None:
    for node in module.body:
        if isinstance(node, ast.Return):
            raise Exception("Main body of function cannot return")


def make_return(var_name: str) -> ast.Return:
    name = ast.Name(id=var_name)
    return ast.Return(value=name)


def make_ast_args(args: list[str]) -> ast.arguments:
    arguments = []
    for arg_name in args:
        arg = ast.arg(arg=arg_name)
        arguments.append(arg)
    return ast.arguments(args=arguments, posonlyargs=[], defaults=[], kwonlyargs=[])


def make_ast_func(
    name: str, input_kwargs: list[str], output_arg: str, body: list[ast.AST]
) -> ast.FunctionDef:
    args = make_ast_args(input_kwargs)
    r = make_return(output_arg)
    new_body = body + [r]
    f = ast.FunctionDef(
        name=name, args=args, body=new_body, decorator_list=[], lineno=0
    )
    return f


def parse_and_wrap_code(
    func_name: str,
    raw_code: str,
    input_kwargs: list[str],
    output_arg: str,
) -> str:
    # convert to AST
    ast_code = ast.parse(raw_code)

    # check there are no globals
    v = GlobalsVisitor()
    v.visit(ast_code)

    # check the main body doesn't return
    check_no_returns(ast_code)

    # wrap user code in a function
    wrapper_function = make_ast_func(
        func_name,
        input_kwargs=input_kwargs,
        output_arg=output_arg,
        body=ast_code.body,
    )

    return unparse(wrapper_function)
