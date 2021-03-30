# stdlib
from typing import Dict
from typing import Union

# from ..misc.union import UnionGenerator

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

allowlist["zksk.expr.Secret"] = "zksk.expr.Secret"
allowlist["zksk.Secret"] = "zksk.expr.Secret"
allowlist["zksk.utils.make_generators"] = "syft.lib.python.List"

allowlist["zksk.primitives.dlrep.DLRep"] = "zksk.primitives.dlrep.DLRep"
allowlist["zksk.DLRep"] = "zksk.primitives.dlrep.DLRep"


allowlist["zksk.expr.Expression"] = "zksk.expr.Expression"
allowlist["zksk.expr.Expression.__getattribute__"] = "zksk.expr.Expression"
allowlist["zksk.expr.Expression.__add__"] = "zksk.expr.Expression"
allowlist["zksk.expr.Expression.__class__"] = "zksk.expr.Expression"
allowlist["zksk.expr.Expression.__delattr__"] = "zksk.expr.Expression"
# allowlist["zksk.expr.Expression.__dict__"] = "dict"
allowlist["zksk.expr.Expression.__dir__"] = "zksk.expr.Expression"
allowlist["zksk.expr.Expression.__doc__"] = "zksk.expr.Expression"
allowlist["zksk.expr.Expression.__eq__"] = "zksk.expr.Expression"
allowlist["zksk.expr.Expression.__format__"] = "zksk.expr.Expression"
allowlist["zksk.expr.Expression.__new__"] = "zksk.expr.Expression"
