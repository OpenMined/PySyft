# stdlib
import ast
import sys
from typing import Any
from typing import List

# relative
from .credentials import SyftVerifyKey
from .document_store import PartitionKey

UserVerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
CodeHashPartitionKey = PartitionKey(key="code_hash", type_=int)

stdout_ = sys.stdout
stderr_ = sys.stderr

PyCodeObject = Any


class GlobalsVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        if isinstance(node, ast.Global):
            raise Exception("No Globals allows")
        ast.NodeVisitor.generic_visit(self, node)


def check_no_returns(module: ast.Module) -> None:
    for node in module.body:
        if isinstance(node, ast.Return):
            raise Exception("Main body of function cannot return")


def make_return(var_names: str) -> ast.Return:
    keys = [ast.Constant(value=var_name) for var_name in var_names]
    values = [ast.Name(id=var_name) for var_name in var_names]
    ast_dict = ast.Dict(keys, values)
    return ast.Return(value=ast_dict)


def make_ast_args(args: List[str]) -> ast.arguments:
    arguments = []
    for arg_name in args:
        arg = ast.arg(arg=arg_name)
        arguments.append(arg)
    return ast.arguments(args=arguments, posonlyargs=[], defaults=[], kwonlyargs=[])


def make_ast_func(
    name: str, input_kwargs: List[str], output_args: List[str], body=List[ast.AST]
) -> ast.FunctionDef:
    args = make_ast_args(input_kwargs)
    r = make_return(output_args)
    new_body = body + [r]
    f = ast.FunctionDef(
        name=name, args=args, body=new_body, decorator_list=[], lineno=0
    )
    return f


def parse_and_wrap_code(
    func_name: str,
    raw_code: str,
    input_kwargs: List[str],
    output_args: List[str],
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
        output_args=output_args,
        body=ast_code.body,
    )

    return ast.unparse(wrapper_function)
