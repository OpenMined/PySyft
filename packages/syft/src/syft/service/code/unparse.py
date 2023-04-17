# stdlib
import _ast
import ast
import sys

# third party
import astunparse  # ast.unparse for python 3.8
from six.moves import cStringIO

# ast.unparse is only in >= python 3.9
# astunparse works on python 3.8 but has a bug caused by
# the ast library in python 3.8 where Constant's dont have a default .kind attribute

# this fixes the bug in ast.unparse https://github.com/IBM/lale/pull/738/files


class FixUnparser(astunparse.Unparser):
    def _Constant(self, t: _ast.expr) -> None:
        if not hasattr(t, "kind"):
            setattr(t, "kind", None)

        super()._Constant(t)


def unparse(tree: _ast.Module) -> str:
    if sys.version_info >= (3, 9):
        return ast.unparse(tree)
    else:
        v = cStringIO()
        FixUnparser(tree, file=v)
        return v.getvalue()
