# stdlib
import ast
import inspect

# third party
from IPython import get_ipython

# relative
from .code_parse import LaunchJobVisitor


def submit_subjobs_code(submit_user_code, ep_client) -> None:  # type: ignore
    # TODO: fix the mypy issue. Reason: circular import
    # We are exploring the source code to automatically upload
    # subjobs in the ephemeral node
    # Usually, a DS would manually submit the code for subjobs,
    # but because we dont allow them to interact with the ephemeral node
    # that would not be possible
    if "domain" in submit_user_code.input_kwargs:
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
