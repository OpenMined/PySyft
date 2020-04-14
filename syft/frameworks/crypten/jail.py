import torch
import crypten
import syft
import re
import ast
from dill.source import getsource
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    guarded_setattr,
)
from RestrictedPython.PrintCollector import PrintCollector


class JailRunner:
    """Execution envrionment with limited capabilities."""

    def __init__(
        self,
        func_src=None,
        func=None,
        modules=[torch, syft, crypten],
        rm_decorators=True,
        **global_kwargs,
    ):
        """
        Control what should be accesible from inside the function.

        Args:
            func_src: function's src to be jailed. Required if func isn't set.
            func: function to be jailed. Ignored if func_src is set.
            modules: python modules that should be accessible.
            rm_decorators: specify if decorators should be removed. Default to True.
            global_kwargs: globals to be accessible.
        """

        if func_src is None:
            if func is None:
                raise ValueError("Either func_src or func must be set")
            else:
                func_src = getsource(func)

        if rm_decorators:
            # Remove decorator if any
            func_src = re.sub(r"@[^\(]+\([^\)]*\)", "", func_src)

        is_func, self._func_name = JailRunner._check_func_def(func_src)
        if not is_func:
            raise ValueError("Not valid function definition")

        self._func_src = func_src

        self._jail_globals = global_kwargs
        # add modules
        for module in modules:
            self._jail_globals[module.__name__] = module

        self._is_built = False
        self._build()

    def _build(self):
        if self._is_built:
            raise RuntimeWarning("JailRunner already built.")

        exec_globals = self._jail_globals
        exec_globals["__builtins__"] = safe_builtins
        exec_globals["_getiter_"] = default_guarded_getiter
        exec_globals["_getitem_"] = default_guarded_getitem
        exec_globals["_getattr_"] = getattr
        # for a, b in
        exec_globals["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        # for a in
        exec_globals["_unpack_sequence_"] = guarded_unpack_sequence
        # unrestricted write of attr
        exec_globals["_write_"] = lambda x: x
        # Collecting printed strings and saved in printed local variable
        exec_globals["_print_"] = PrintCollector
        # exec_globals["__name__"] = "__main__"

        exec_locals = {}
        compiled = compile_restricted(self._func_src)
        exec(compiled, exec_globals, exec_locals)  # nosec
        self._func = exec_locals[self._func_name]

    def run(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @classmethod
    def _check_func_def(cls, func_src):
        """Check if the func_src is really a function definition.
        Returns a boolean is_func and the name of the function if is_func is True.
        """

        # The body should contain one element
        tree = ast.parse(func_src)
        if len(tree.body) != 1:
            return (False, "")
        # The body should contain a function defintion
        func_def = tree.body[0]
        if type(func_def) is not ast.FunctionDef:
            return (False, "")

        return (True, func_def.name)
