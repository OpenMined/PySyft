# stdlib
import ast
import inspect
from inspect import Signature
from typing import Any
from typing import Callable

# relative
from ...serde.serializable import serializable
from ...serde.signature import signature_remove_context
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ..context import AuthedServiceContext
from ..response import SyftError


@serializable()
class CustomAPIEndpoint(SyftObject):
    # version
    __canonical_name__ = "CustomAPIEndpoint"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    api_code: str
    signature: Signature
    func_name: str

    __attr_searchable__ = ["path"]
    __attr_unique__ = ["path"]

    def exec(self, context: AuthedServiceContext, **kwargs: Any) -> Any:
        try:
            inner_function = ast.parse(self.api_code).body[0]
            inner_function.decorator_list = []
            # compile the function
            raw_byte_code = compile(ast.unparse(inner_function), "<string>", "exec")
            # load it
            exec(raw_byte_code)  # nosec
            # execute it
            evil_string = f"{self.func_name}(context, **kwargs)"
            result = eval(evil_string, None, locals())  # nosec
            # return the results
            return context, result
        except Exception as e:
            print(f"Failed to run CustomAPIEndpoint Code. {e}")
            return SyftError(e)


def get_signature(func: Callable) -> Signature:
    sig = inspect.signature(func)
    sig = signature_remove_context(sig)
    return sig


def api_endpoint(path: str) -> CustomAPIEndpoint:
    def decorator(f):
        res = CustomAPIEndpoint(
            path=path,
            api_code=inspect.getsource(f),
            signature=get_signature(f),
            func_name=f.__name__,
        )
        return res

    return decorator
