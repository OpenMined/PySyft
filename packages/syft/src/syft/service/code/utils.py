# stdlib
import ast
import inspect

# third party
from IPython import get_ipython

# relative
from ...types.errors import SyftException
from ..response import SyftWarning
from .code_parse import GlobalsVisitor
from .code_parse import LaunchJobVisitor


def submit_subjobs_code(submit_user_code, ep_client) -> None:  # type: ignore
    # TODO: fix the mypy issue. Reason: circular import
    # We are exploring the source code to automatically upload
    # subjobs in the ephemeral server
    # Usually, a DS would manually submit the code for subjobs,
    # but because we dont allow them to interact with the ephemeral server
    # that would not be possible
    if "datasite" in submit_user_code.input_kwargs:
        tree = ast.parse(inspect.getsource(submit_user_code.local_function))
        v = LaunchJobVisitor()
        v.visit(tree)
        nested_calls = v.nested_calls
        try:
            ipython = (
                get_ipython()
            )  # works only in interactive envs (like jupyter notebooks)
        except Exception:
            ipython = None
            pass

        for call in nested_calls:
            if ipython is not None:
                specs = ipython.object_inspect(call)
                # Look for nested job locally, maybe we could
                # fetch
                if specs["type_name"] == "SubmitUserCode":
                    ep_client.code.submit(ipython.ev(call))


def check_for_global_vars(code_tree: ast.Module) -> GlobalsVisitor | SyftWarning:
    """
    Check that the code does not contain any global variables
    """
    v = GlobalsVisitor()
    try:
        v.visit(code_tree)
    except Exception:
        raise SyftException(
            public_message="Your code contains (a) global variable(s), which is not allowed"
        )
    return v


def parse_code(raw_code: str) -> ast.Module | SyftWarning:
    """
    Parse the code into an AST tree and return a warning if there are syntax errors
    """
    try:
        tree = ast.parse(raw_code)
    except SyntaxError as e:
        raise SyftException(public_message=f"Your code contains syntax error: {e}")
    return tree
